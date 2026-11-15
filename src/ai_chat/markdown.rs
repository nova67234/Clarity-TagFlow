//! What a reply's text needs before it's drawn as markdown — the things
//! models write that markdown reads another way:
//!
//! - prices (`$5-$10`), which the maths extension takes for a formula;
//! - maths in `\( … \)` and `\[ … \]`, which it doesn't take for one;
//! - the typing cursor, which mustn't land on a code block's closing fence.
//!
//! And the maths itself: TeX to plain Unicode ([`tex_to_unicode`]) — there's
//! no maths engine, but what local models write is mostly simple enough to
//! read well that way (`\frac{a}{b}` → a/b, `x^{2}` → x², `\leq` → ≤).

use std::borrow::Cow;
use std::ops::Range;

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

/// The viewer's own set (egui_commonmark's), so a reply is read here as it
/// will be drawn.
fn options(math: bool) -> Options {
    let o = Options::ENABLE_TABLES
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_DEFINITION_LIST;
    if math { o | Options::ENABLE_MATH } else { o }
}

/// A reply's text as it's handed to the markdown viewer. (Most replies have
/// none of this in them and are handed over as they are.)
pub(super) fn prepare(text: &str) -> Cow<'_, str> {
    let delimiters = text.contains("\\(") || text.contains("\\[");
    if !delimiters && !text.contains('$') {
        return Cow::Borrowed(text);
    }
    let text = if delimiters { tex_delimiters(text) } else { Cow::Borrowed(text) };
    match currency_dollars(&text) {
        Some(fixed) => Cow::Owned(fixed),
        None => text,
    }
}

/// Where the code is (spans and blocks): nothing in it is touched.
fn code_ranges(text: &str) -> Vec<Range<usize>> {
    let mut code = Vec::new();
    let mut open = None;
    for (event, range) in Parser::new_ext(text, options(false)).into_offset_iter() {
        match event {
            Event::Code(_) => code.push(range),
            Event::Start(Tag::CodeBlock(_)) => open = Some(range.start),
            Event::End(TagEnd::CodeBlock) => {
                if let Some(start) = open.take() {
                    code.push(start..range.end);
                }
            }
            _ => {}
        }
    }
    code
}

/// `\( … \)` → `$…$` and `\[ … \]` → `$$…$$`: how many models write their
/// maths, and what markdown reads as escaped brackets around ordinary text.
/// Only where what's inside looks like maths — `\[1\]` is a citation someone
/// escaped, and stays one.
fn tex_delimiters(text: &str) -> Cow<'_, str> {
    let code = code_ranges(text);
    let in_code = |at: usize| code.iter().any(|r| r.contains(&at));
    let bytes = text.as_bytes();
    let mut out = String::new();
    let mut copied = 0;
    let mut i = 0;
    while i + 1 < bytes.len() {
        let (close, mark) = match (bytes[i], bytes[i + 1]) {
            (b'\\', b'(') => ("\\)", "$"),
            (b'\\', b'[') => ("\\]", "$$"),
            // (An escaped backslash, then whatever's next.)
            (b'\\', b'\\') => {
                i += 2;
                continue;
            }
            _ => {
                i += 1;
                continue;
            }
        };
        let inner_start = i + 2;
        // It closes in the same paragraph, or it isn't maths.
        let paragraph_end = text[inner_start..].find("\n\n").map_or(text.len(), |p| inner_start + p);
        let closed = text[inner_start..paragraph_end].find(close).map(|p| inner_start + p);
        match closed {
            Some(end) if !in_code(i) && looks_like_maths(&text[inner_start..end]) => {
                out.push_str(&text[copied..i]);
                out.push_str(mark);
                // (One line, tight to its marks: the extension wants no space
                // inside them.)
                out.push_str(&text[inner_start..end].split_whitespace().collect::<Vec<_>>().join(" "));
                out.push_str(mark);
                copied = end + 2;
                i = copied;
            }
            _ => i += 2,
        }
    }
    if copied == 0 {
        return Cow::Borrowed(text);
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

/// What's between `\(` and `\)` is maths, not escaped brackets round words
/// or a number: it has an operator or a command in it, or it's a short name
/// (`x`, `2n`).
fn looks_like_maths(inner: &str) -> bool {
    let inner = inner.trim();
    if inner.is_empty() || inner.contains('$') {
        return false;
    }
    // (A citation: `\[1\]`, `\[2, 3\]`, `\[4-6\]`.)
    if inner.chars().all(|c| c.is_ascii_digit() || matches!(c, ',' | ' ' | '-' | '–')) {
        return false;
    }
    inner.contains(['\\', '=', '^', '_', '+', '<', '>', '*', '/', '|', '-'])
        || (inner.chars().count() <= 3 && inner.chars().all(char::is_alphanumeric))
}

/// Prices the maths extension took for a formula, their dollar signs
/// escaped: "$5-$10" was read as the formula "5-" and lost both signs. The
/// tell is pandoc's rule for the same trouble — a formula's closing `$` has
/// no digit straight after it. `None` when there are none.
fn currency_dollars(text: &str) -> Option<String> {
    if !text.contains('$') {
        return None;
    }
    let bytes = text.as_bytes();
    let mut signs: Vec<usize> = Vec::new();
    for (event, range) in Parser::new_ext(text, options(true)).into_offset_iter() {
        if matches!(event, Event::InlineMath(_))
            && bytes.get(range.end).is_some_and(u8::is_ascii_digit)
            && bytes[range.start] == b'$'
            && bytes[range.end - 1] == b'$'
        {
            signs.extend([range.start, range.end - 1]);
        }
    }
    if signs.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(text.len() + signs.len());
    let mut copied = 0;
    for at in signs {
        out.push_str(&text[copied..at]);
        out.push('\\');
        copied = at;
    }
    out.push_str(&text[copied..]);
    Some(out)
}

/// Put the typing cursor on the end of a reply that's still coming in. After
/// a code block's closing fence it goes under the block: on the fence's own
/// line it made the fence part of the code ("```▌"), until more came in.
pub(super) fn push_cursor(text: &mut String) {
    if ends_on_closing_fence(text) {
        text.push_str("\n\n");
    }
    text.push('▌');
}

/// `text`'s last line is the closing fence of a code block.
fn ends_on_closing_fence(text: &str) -> bool {
    // The fence a block was opened with: its character, and how many.
    let mut open: Option<(char, usize)> = None;
    let mut closed_here = false;
    for line in text.lines() {
        closed_here = false;
        // (In a list or a quote it's indented, or behind `>`s.)
        let line = line.trim_start_matches([' ', '\t', '>']);
        let Some(mark) = line.chars().next().filter(|c| matches!(c, '`' | '~')) else { continue };
        let run = line.chars().take_while(|c| *c == mark).count();
        if run < 3 {
            continue;
        }
        match open {
            None => open = Some((mark, run)),
            Some((with, at_least)) => {
                if mark == with && run >= at_least && line[run..].trim().is_empty() {
                    open = None;
                    closed_here = true;
                }
            }
        }
    }
    closed_here && !text.ends_with('\n')
}

// --- Maths: TeX to Unicode. ---

/// What TeX commands become: the symbol, and what to write where the fonts
/// here can't show it ("" for the command's own name, as an unknown one is
/// written: `\sin` is "sin").
const SYMBOLS: &[(&str, &str, &str)] = &[
    // Greek.
    ("alpha", "α", ""), ("beta", "β", ""), ("gamma", "γ", ""), ("delta", "δ", ""),
    ("epsilon", "ε", ""), ("varepsilon", "ε", ""), ("zeta", "ζ", ""), ("eta", "η", ""),
    ("theta", "θ", ""), ("vartheta", "ϑ", "theta"), ("iota", "ι", ""), ("kappa", "κ", ""),
    ("lambda", "λ", ""), ("mu", "μ", ""), ("nu", "ν", ""), ("xi", "ξ", ""),
    ("omicron", "ο", ""), ("pi", "π", ""), ("varpi", "ϖ", "pi"), ("rho", "ρ", ""),
    ("varrho", "ϱ", "rho"), ("sigma", "σ", ""), ("varsigma", "ς", "sigma"), ("tau", "τ", ""),
    ("upsilon", "υ", ""), ("phi", "φ", ""), ("varphi", "φ", ""), ("chi", "χ", ""),
    ("psi", "ψ", ""), ("omega", "ω", ""),
    ("Gamma", "Γ", ""), ("Delta", "Δ", ""), ("Theta", "Θ", ""), ("Lambda", "Λ", ""),
    ("Xi", "Ξ", ""), ("Pi", "Π", ""), ("Sigma", "Σ", ""), ("Upsilon", "Υ", ""),
    ("Phi", "Φ", ""), ("Psi", "Ψ", ""), ("Omega", "Ω", ""),
    // Operators.
    ("times", "×", "x"), ("cdot", "·", "*"), ("div", "÷", "/"), ("pm", "±", "+/-"),
    ("mp", "∓", "-/+"), ("ast", "∗", "*"), ("star", "⋆", "*"), ("circ", "∘", "o"),
    ("bullet", "•", "*"), ("oplus", "⊕", "(+)"), ("otimes", "⊗", "(x)"), ("odot", "⊙", "(.)"),
    ("setminus", "∖", "\\"), ("bmod", "mod", ""), ("mod", "mod", ""),
    // Relations.
    ("leq", "≤", "<="), ("le", "≤", "<="), ("leqslant", "≤", "<="),
    ("geq", "≥", ">="), ("ge", "≥", ">="), ("geqslant", "≥", ">="),
    ("neq", "≠", "!="), ("ne", "≠", "!="), ("approx", "≈", "~="), ("equiv", "≡", "=="),
    ("sim", "∼", "~"), ("simeq", "≃", "~="), ("cong", "≅", "~="), ("propto", "∝", ""),
    ("ll", "≪", "<<"), ("gg", "≫", ">>"), ("lt", "<", ""), ("gt", ">", ""), ("doteq", "≐", "="),
    // Sets and logic.
    ("in", "∈", ""), ("notin", "∉", "not in"), ("ni", "∋", ""),
    ("subset", "⊂", ""), ("supset", "⊃", ""), ("subseteq", "⊆", ""), ("supseteq", "⊇", ""),
    ("cup", "∪", ""), ("cap", "∩", ""), ("emptyset", "∅", ""), ("varnothing", "∅", "emptyset"),
    ("forall", "∀", ""), ("exists", "∃", ""), ("nexists", "∄", "not exists"),
    ("neg", "¬", "not "), ("lnot", "¬", "not "), ("land", "∧", "and"), ("wedge", "∧", "and"),
    ("lor", "∨", "or"), ("vee", "∨", "or"), ("therefore", "∴", ""), ("because", "∵", ""),
    ("top", "⊤", ""), ("bot", "⊥", ""), ("perp", "⊥", ""), ("parallel", "∥", "||"),
    ("mid", "|", ""), ("vert", "|", ""), ("lvert", "|", ""), ("rvert", "|", ""),
    ("Vert", "‖", "||"), ("lVert", "‖", "||"), ("rVert", "‖", "||"),
    // Sums and calculus.
    ("sum", "∑", ""), ("prod", "∏", ""), ("int", "∫", ""), ("iint", "∬", "int int"),
    ("iiint", "∭", "int int int"), ("oint", "∮", "int"), ("partial", "∂", "d"),
    ("nabla", "∇", ""), ("infty", "∞", "inf"),
    // Arrows.
    ("to", "→", "->"), ("rightarrow", "→", "->"), ("longrightarrow", "→", "->"),
    ("gets", "←", "<-"), ("leftarrow", "←", "<-"), ("longleftarrow", "←", "<-"),
    ("leftrightarrow", "↔", "<->"), ("longleftrightarrow", "↔", "<->"),
    ("Rightarrow", "⇒", "=>"), ("implies", "⇒", "=>"), ("Longrightarrow", "⇒", "=>"),
    ("Leftarrow", "⇐", "<="), ("impliedby", "⇐", "<="), ("Longleftarrow", "⇐", "<="),
    ("Leftrightarrow", "⇔", "<=>"), ("iff", "⇔", "<=>"), ("Longleftrightarrow", "⇔", "<=>"),
    ("mapsto", "↦", "|->"), ("uparrow", "↑", "^"), ("downarrow", "↓", "v"),
    ("Uparrow", "⇑", "^"), ("Downarrow", "⇓", "v"), ("nearrow", "↗", ""), ("searrow", "↘", ""),
    ("rightleftharpoons", "⇌", "<=>"),
    // The rest.
    ("degree", "°", " deg"), ("angle", "∠", ""), ("triangle", "△", ""), ("square", "□", ""),
    ("prime", "′", "'"), ("hbar", "ℏ", "h"), ("ell", "ℓ", "l"), ("Re", "ℜ", ""), ("Im", "ℑ", ""),
    ("aleph", "ℵ", ""), ("dagger", "†", "+"), ("ddagger", "‡", "++"),
    ("ldots", "…", "..."), ("dots", "…", "..."), ("dotsc", "…", "..."), ("cdots", "⋯", "..."),
    ("dotsb", "⋯", "..."), ("vdots", "⋮", "..."), ("ddots", "⋱", "..."),
    ("langle", "⟨", "<"), ("rangle", "⟩", ">"), ("lfloor", "⌊", "floor("), ("rfloor", "⌋", ")"),
    ("lceil", "⌈", "ceil("), ("rceil", "⌉", ")"), ("lbrace", "{", ""), ("rbrace", "}", ""),
    ("lbrack", "[", ""), ("rbrack", "]", ""), ("backslash", "\\", ""), ("checkmark", "✓", ""),
    ("quad", "  ", ""), ("qquad", "    ", ""),
];

/// Symbols that sit against the letter after them, as the Greek ones do.
const TIGHT: &[&str] = &["partial", "nabla", "neg", "lnot", "hbar", "ell", "forall", "exists", "nexists", "angle"];

/// Commands that say how to set what follows and say nothing themselves.
const SILENT: &[&str] = &[
    "left", "right", "middle", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
    "Biggl", "Biggr", "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle", "limits", "nolimits",
    "nonumber", "notag", "mathstrut", "strut",
];

/// Commands that only set their argument in another face: the argument, as
/// it is.
const FACES: &[&str] = &[
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathscr", "mathfrak", "boldsymbol", "bm",
    "operatorname", "underline", "boxed", "mathbin", "mathrel", "mathop", "mathord", "overbrace", "underbrace",
    "widehat", "widetilde", "overrightarrow",
];

/// Commands whose argument is words, not maths.
const WORDS: &[&str] = &["text", "textrm", "textbf", "textit", "textsf", "texttt", "textnormal", "mbox", "hbox"];

/// Raised and lowered characters there are Unicode ones for.
const RAISED: &[(char, char)] = &[
    ('0', '⁰'), ('1', '¹'), ('2', '²'), ('3', '³'), ('4', '⁴'), ('5', '⁵'), ('6', '⁶'), ('7', '⁷'),
    ('8', '⁸'), ('9', '⁹'), ('+', '⁺'), ('-', '⁻'), ('=', '⁼'), ('(', '⁽'), (')', '⁾'),
    ('a', 'ᵃ'), ('b', 'ᵇ'), ('c', 'ᶜ'), ('d', 'ᵈ'), ('e', 'ᵉ'), ('f', 'ᶠ'), ('g', 'ᵍ'), ('h', 'ʰ'),
    ('i', 'ⁱ'), ('j', 'ʲ'), ('k', 'ᵏ'), ('l', 'ˡ'), ('m', 'ᵐ'), ('n', 'ⁿ'), ('o', 'ᵒ'), ('p', 'ᵖ'),
    ('r', 'ʳ'), ('s', 'ˢ'), ('t', 'ᵗ'), ('u', 'ᵘ'), ('v', 'ᵛ'), ('w', 'ʷ'), ('x', 'ˣ'), ('y', 'ʸ'),
    ('z', 'ᶻ'), ('T', 'ᵀ'),
];
const LOWERED: &[(char, char)] = &[
    ('0', '₀'), ('1', '₁'), ('2', '₂'), ('3', '₃'), ('4', '₄'), ('5', '₅'), ('6', '₆'), ('7', '₇'),
    ('8', '₈'), ('9', '₉'), ('+', '₊'), ('-', '₋'), ('=', '₌'), ('(', '₍'), (')', '₎'),
    ('a', 'ₐ'), ('e', 'ₑ'), ('h', 'ₕ'), ('i', 'ᵢ'), ('j', 'ⱼ'), ('k', 'ₖ'), ('l', 'ₗ'), ('m', 'ₘ'),
    ('n', 'ₙ'), ('o', 'ₒ'), ('p', 'ₚ'), ('r', 'ᵣ'), ('s', 'ₛ'), ('t', 'ₜ'), ('u', 'ᵤ'), ('v', 'ᵥ'),
    ('x', 'ₓ'),
];

/// TeX maths as plain Unicode text. `can_show` says whether the fonts here
/// have a character: what they haven't is written another way (`x^n` rather
/// than a box where ⁿ would be).
pub(super) fn tex_to_unicode(tex: &str, can_show: &mut dyn FnMut(char) -> bool) -> String {
    let mut reader = Tex { chars: tex.chars().collect(), at: 0, can_show };
    let out = reader.sequence(None);
    // (TeX's spacing is its own: runs of spaces are one, as they'd be there.)
    out.lines().map(|line| line.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join("\n")
}

struct Tex<'a> {
    chars: Vec<char>,
    at: usize,
    can_show: &'a mut dyn FnMut(char) -> bool,
}

impl Tex<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn skip_spaces(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.at += 1;
        }
    }

    /// Everything up to `end` (which is taken), or to the end.
    fn sequence(&mut self, end: Option<char>) -> String {
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if Some(c) == end {
                self.at += 1;
                break;
            }
            self.atom(&mut out);
        }
        out
    }

    /// A command's argument: a braced group, or the one thing that's next.
    fn argument(&mut self) -> String {
        self.skip_spaces();
        let mut out = String::new();
        match self.peek() {
            Some('{') => {
                self.at += 1;
                out = self.sequence(Some('}'));
            }
            Some(_) => self.atom(&mut out),
            None => {}
        }
        out
    }

    /// An argument that's words (`\text{ if }`): as written, spaces and all.
    fn words(&mut self) -> String {
        self.skip_spaces();
        if self.peek() != Some('{') {
            return self.argument();
        }
        self.at += 1;
        let mut out = String::new();
        let mut depth = 0;
        while let Some(c) = self.peek() {
            self.at += 1;
            match c {
                '\\' => {
                    // (`\}`, `\%`: the character itself.)
                    if let Some(next) = self.peek() {
                        self.at += 1;
                        out.push(next);
                    }
                }
                '{' => depth += 1,
                '}' if depth == 0 => break,
                '}' => depth -= 1,
                c => out.push(c),
            }
        }
        out
    }

    fn atom(&mut self, out: &mut String) {
        let Some(c) = self.peek() else { return };
        self.at += 1;
        match c {
            '\\' => self.command(out),
            '{' => out.push_str(&self.sequence(Some('}'))),
            '}' => {}
            '^' => {
                let from = self.at;
                let raised = self.argument();
                // 90^\circ is 90°.
                let source: String =
                    self.chars[from..self.at].iter().filter(|c| !matches!(c, '{' | '}' | ' ')).collect();
                if source == "\\circ" {
                    out.push_str(self.shown("°", " deg"));
                } else {
                    out.push_str(&self.script(&raised, true));
                }
            }
            '_' => {
                let lowered = self.argument();
                out.push_str(&self.script(&lowered, false));
            }
            // (A column of an aligned environment; a tie.)
            '&' => {}
            '~' => out.push(' '),
            c => out.push(c),
        }
    }

    /// `symbol` where the fonts can show all of it, else `plain`.
    fn shown<'s>(&mut self, symbol: &'s str, plain: &'s str) -> &'s str {
        if symbol.chars().all(|c| c.is_ascii() || (self.can_show)(c)) { symbol } else { plain }
    }

    /// The command whose backslash was just read.
    fn command(&mut self, out: &mut String) {
        let Some(first) = self.peek() else { return };
        if !first.is_ascii_alphabetic() {
            self.at += 1;
            match first {
                ',' | ';' | ':' | ' ' | '>' => out.push(' '),
                '!' => {}
                '\\' => out.push('\n'),
                '|' => out.push_str(self.shown("‖", "||")),
                // (`\{`, `\%`, `\$`, `\_`: the character itself.)
                c => out.push(c),
            }
            return;
        }
        let mut name = String::new();
        while let Some(c) = self.peek().filter(char::is_ascii_alphabetic) {
            name.push(c);
            self.at += 1;
        }
        match name.as_str() {
            "frac" | "dfrac" | "tfrac" | "cfrac" => {
                let (over, under) = (self.argument(), self.argument());
                out.push_str(&format!("{}/{}", grouped(&over), grouped(&under)));
            }
            "binom" | "dbinom" | "tbinom" => {
                let (n, k) = (self.argument(), self.argument());
                out.push_str(&format!("C({}, {})", n.trim(), k.trim()));
            }
            "sqrt" => {
                let root = (self.peek() == Some('[')).then(|| {
                    self.at += 1;
                    self.sequence(Some(']'))
                });
                let of = self.argument();
                let sign = match root.as_deref().map(str::trim) {
                    None | Some("2") => self.shown("√", "sqrt").to_string(),
                    Some("3") if (self.can_show)('∛') => "∛".to_string(),
                    Some("4") if (self.can_show)('∜') => "∜".to_string(),
                    Some(n) => format!("{}{}", self.script(n, true), self.shown("√", "sqrt")),
                };
                out.push_str(&sign);
                // (Spelt out, it's a function: sqrt(x).)
                if sign.ends_with("sqrt") {
                    out.push_str(&format!("({})", of.trim()));
                } else {
                    out.push_str(&grouped(&of));
                }
            }
            "mathbb" => {
                let letters = self.argument();
                for c in letters.chars() {
                    let double = match c {
                        'R' => 'ℝ',
                        'N' => 'ℕ',
                        'Z' => 'ℤ',
                        'Q' => 'ℚ',
                        'C' => 'ℂ',
                        'P' => 'ℙ',
                        c => c,
                    };
                    out.push(if (self.can_show)(double) { double } else { c });
                }
            }
            "hat" | "bar" | "overline" | "vec" | "tilde" | "dot" | "ddot" => {
                let over = self.argument();
                let mark = match name.as_str() {
                    "hat" => '\u{302}',
                    "bar" | "overline" => '\u{304}',
                    "vec" => '\u{20d7}',
                    "tilde" => '\u{303}',
                    "dot" => '\u{307}',
                    _ => '\u{308}',
                };
                out.push_str(&over);
                // (Over one letter. Over more there's no writing it.)
                if over.chars().count() == 1 && (self.can_show)(mark) {
                    out.push(mark);
                }
            }
            "not" => {
                self.skip_spaces();
                let mut next = String::new();
                self.atom(&mut next);
                let crossed = match next.as_str() {
                    "=" => "≠",
                    "∈" => "∉",
                    "⊂" => "⊄",
                    "≡" => "≢",
                    "∃" => "∄",
                    "<" => "≮",
                    ">" => "≯",
                    _ => "",
                };
                if !crossed.is_empty() && crossed.chars().all(|c| (self.can_show)(c)) {
                    out.push_str(crossed);
                } else {
                    out.push_str("not ");
                    out.push_str(&next);
                }
            }
            "pmod" => {
                let of = self.argument();
                out.push_str(&format!(" (mod {})", of.trim()));
            }
            // An environment's name says nothing: its rows are what's read.
            "begin" | "end" => {
                self.words();
            }
            name if WORDS.contains(&name) => out.push_str(&self.words()),
            name if FACES.contains(&name) => out.push_str(&self.argument()),
            name if SILENT.contains(&name) => {
                // (`\left.` and `\right.` are "no bracket here".)
                if matches!(name, "left" | "right") && self.peek() == Some('.') {
                    self.at += 1;
                }
            }
            name => match SYMBOLS.iter().find(|(command, _, _)| *command == name) {
                Some((_, symbol, plain)) => {
                    let plain = if plain.is_empty() { name } else { plain };
                    let shown = self.shown(symbol, plain).to_string();
                    out.push_str(&shown);
                    // The space that only ends a letter's name isn't one in
                    // the maths: `2\pi r` is 2πr, `\partial f` is ∂f. (An
                    // operator keeps its spaces: a × b.)
                    let letter = symbol.chars().all(|c| ('\u{370}'..='\u{3ff}').contains(&c)) || TIGHT.contains(&name);
                    if letter && shown == *symbol {
                        let from = self.at;
                        self.skip_spaces();
                        if !self.peek().is_some_and(char::is_alphanumeric) {
                            self.at = from;
                        }
                    }
                }
                // (`\sin`, `\log`, `\max`: their names.)
                None => out.push_str(name),
            },
        }
    }

    /// `text` raised (`^`) or lowered (`_`): in the Unicode characters for
    /// it where there are ones for all of it, else marked as TeX marks it.
    fn script(&mut self, text: &str, raised: bool) -> String {
        let text = text.trim();
        if text.is_empty() {
            return String::new();
        }
        // (A prime or a star is raised already.)
        if raised && text.chars().all(|c| matches!(c, '′' | '\'' | '*')) {
            return text.to_string();
        }
        let table = if raised { RAISED } else { LOWERED };
        let unicode: Option<String> = text
            .chars()
            .map(|c| table.iter().find(|(plain, _)| *plain == c).map(|(_, to)| *to).filter(|to| (self.can_show)(*to)))
            .collect();
        let mark = if raised { '^' } else { '_' };
        match unicode {
            Some(unicode) => unicode,
            None if text.chars().count() == 1 => format!("{mark}{text}"),
            None => format!("{mark}({text})"),
        }
    }
}

/// `part` of a fraction (or under a root), in brackets where it's more than
/// one term: (n+1)/2, but a/b.
fn grouped(part: &str) -> String {
    let part = part.trim();
    let terms = part.contains(|c: char| c.is_whitespace() || "+-−±*/·×=<>,".contains(c));
    if terms { format!("({part})") } else { part.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex(source: &str) -> String {
        tex_to_unicode(source, &mut |_| true)
    }

    /// The commands local models write come out as the maths reads.
    #[test]
    fn tex_reads_as_the_maths_does() {
        for (source, shown) in [
            // (What the old table made of these: "\fracab", "·s", "\textdistance",
            // "rightarrow", "≠g p", "90^∘".)
            (r"\frac{a}{b}", "a/b"),
            (r"\frac{n(n+1)}{2}", "(n(n+1))/2"),
            (r"\frac{\partial f}{\partial x}", "∂f/∂x"),
            (r"\frac 1 2", "1/2"),
            (r"1, 2, \cdots, n", "1, 2, ⋯, n"),
            (r"v = \text{distance} / \text{time}", "v = distance / time"),
            (r"f(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}", "f(x) = 1 if x > 0\n0 otherwise"),
            (r"\mathbf{x} \cdot \mathbf{y}", "x · y"),
            (r"p \leftrightarrow q", "p ↔ q"),
            (r"a \Leftrightarrow b", "a ⇔ b"),
            (r"\neg p, \neg (p \land q)", "¬p, ¬ (p ∧ q)"),
            (r"2\pi r, \pi r^2, \theta = 5, \alpha \beta", "2πr, πr², θ = 5, α β"),
            (r"90^\circ", "90°"),
            (r"90^{\circ}", "90°"),
            (r"x^{2} + y_i", "x² + yᵢ"),
            (r"x^2 + x^{10} + e^{-x}", "x² + x¹⁰ + e⁻ˣ"),
            (r"a_1, a_{12}, a_{n+1}", "a₁, a₁₂, aₙ₊₁"),
            (r"x^{a+b}_q, x^{A+B}, x^q", "xᵃ⁺ᵇ_q, x^(A+B), x^q"),
            (r"f'(x), f^\prime, x^*", "f'(x), f′, x*"),
            (r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}", "∑ᵢ₌₁ⁿ i = (n(n+1))/2"),
            (r"\sqrt{2}, \sqrt{x+1}, \sqrt[3]{x}, \sqrt[n]{x}", "√2, √(x+1), ∛x, ⁿ√x"),
            (r"\left( \frac{a}{b} \right)^2", "( a/b )²"),
            (r"\int_0^\infty e^{-x}\,dx", "∫₀^∞ e⁻ˣ dx"),
            (r"x \in \mathbb{R}, n \notin \mathbb{N}", "x ∈ ℝ, n ∉ ℕ"),
            (r"a \not= b, a \not\in B, a \ne b", "a ≠ b, a ∉ B, a ≠ b"),
            (r"\sin^2\theta + \cos^2\theta = 1", "sin²θ + cos²θ = 1"),
            (r"\lim_{x \to 0} \log x", "lim_(x → 0) log x"),
            (r"\binom{n}{k}", "C(n, k)"),
            (r"\hat{x}, \vec{v}, \bar{x}", "x\u{302}, v\u{20d7}, x\u{304}"),
            (r"\{a, b\} \subseteq S, 50\%", "{a, b} ⊆ S, 50%"),
            (r"a \leq b \le c \geq d \ge e", "a ≤ b ≤ c ≥ d ≥ e"),
            (r"\int \in \infty \to \top", "∫ ∈ ∞ → ⊤"),
            (r"a \quad b", "a b"),
            (r"E = mc^2", "E = mc²"),
            (r"\alpha\beta\Gamma", "αβΓ"),
            (r"\unknowncommand{x}", "unknowncommandx"),
            (r"x^", "x"),
            (r"\", ""),
            (r"{a", "a"),
        ] {
            assert_eq!(tex(source), shown, "{source}");
        }
    }

    /// What the fonts can't show is written another way, not as a box.
    #[test]
    fn tex_keeps_to_what_the_fonts_have() {
        let mut ascii_only = |c: char| c.is_ascii();
        for (source, shown) in [
            (r"x^{2} + y_i", "x^2 + y_i"),
            (r"x^{10}", "x^(10)"),
            (r"a \leq b \Rightarrow c \to d", "a <= b => c -> d"),
            (r"\alpha \times \infty", "alpha x inf"),
            (r"90^\circ", "90 deg"),
            (r"\sqrt{x+1}, \sqrt[3]{x}", "sqrt(x+1), ^3sqrt(x)"),
            (r"x \in \mathbb{R}", "x in R"),
            (r"a \not= b, \hat{x}", "a not = b, x"),
            (r"\neg p, \alpha x", "not p, alpha x"),
        ] {
            assert_eq!(tex_to_unicode(source, &mut ascii_only), shown, "{source}");
        }
        // One it has and one it hasn't: each on its own.
        let mut no_small_i = |c: char| c != 'ᵢ';
        assert_eq!(tex_to_unicode("x_i + x_1", &mut no_small_i), "x_i + x₁");
    }

    /// Prices keep their dollar signs: "$5-$10" was read as the formula "5-".
    #[test]
    fn prices_are_not_maths() {
        for (text, prepared) in [
            ("It costs $5-$10 per month.", r"It costs \$5-\$10 per month."),
            ("$1,000-$2,000, or $5/$10/$15 each", r"\$1,000-\$2,000, or \$5/\$10/$15 each"),
            ("**$5**-**$10**", r"**\$5**-**\$10**"),
            // These were never taken for maths.
            ("It costs $20 to $30 per year.", "It costs $20 to $30 per year."),
            ("Just $5.", "Just $5."),
            // And maths stays maths.
            ("So $x^2$ and $a$-$b$ and $$E = mc^2$$.", "So $x^2$ and $a$-$b$ and $$E = mc^2$$."),
            // (Code is as it's written.)
            ("Run `echo $5-$10` now", "Run `echo $5-$10` now"),
            ("```sh\necho $5-$10\n```", "```sh\necho $5-$10\n```"),
        ] {
            assert_eq!(prepare(text), prepared, "{text}");
        }
        // What the viewer makes of the first: no formula in it, both signs.
        let prepared = prepare("It costs $5-$10 per month.");
        let events: Vec<Event> = Parser::new_ext(&prepared, options(true)).collect();
        assert!(!events.iter().any(|e| matches!(e, Event::InlineMath(_))), "{events:?}");
        let text: String = events.iter().filter_map(|e| if let Event::Text(t) = e { Some(t.as_ref()) } else { None }).collect();
        assert_eq!(text, "It costs $5-$10 per month.");
    }

    /// `\( … \)` and `\[ … \]` are maths, as `$ … $` and `$$ … $$` are.
    #[test]
    fn bracketed_maths_is_maths() {
        for (text, prepared) in [
            (r"Inline \( x^2 \) and \(n\).", "Inline $x^2$ and $n$."),
            ("Display:\n\\[ E = mc^2 \\]", "Display:\n$$E = mc^2$$"),
            ("\\[\n\\frac{a}{b}\n= c\n\\]", r"$$\frac{a}{b} = c$$"),
            // Escaped brackets round a citation, or round words, are those.
            (r"As shown \[1\] and \[2, 3\].", r"As shown \[1\] and \[2, 3\]."),
            (r"An \(optional\) word.", r"An \(optional\) word."),
            // One that never closes, or closes a paragraph on.
            (r"A \( b", r"A \( b"),
            ("A \\( x = 1\n\nB \\) c", "A \\( x = 1\n\nB \\) c"),
            // An escaped backslash before a bracket isn't one of these.
            (r"a \\( x = 1 \\)", r"a \\( x = 1 \\)"),
            // (Code is as it's written.)
            (r"Write `\( x^2 \)` for maths", r"Write `\( x^2 \)` for maths"),
            ("```tex\n\\[ x = 1 \\]\n```", "```tex\n\\[ x = 1 \\]\n```"),
            // A price inside one: still a price after.
            (r"\( a = b \) costs $5-$10", r"$a = b$ costs \$5-\$10"),
        ] {
            assert_eq!(prepare(text), prepared, "{text}");
        }
        let maths = |text: &str| -> Vec<String> {
            Parser::new_ext(&prepare(text), options(true))
                .filter_map(|e| match e {
                    Event::InlineMath(m) => Some(format!("inline {m}")),
                    Event::DisplayMath(m) => Some(format!("display {m}")),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(maths(r"Inline \( x^2 \) and display: \[ E = mc^2 \]"), ["inline x^2", "display E = mc^2"]);
    }

    /// A reply with none of it is handed over as it is (no copy made).
    #[test]
    fn most_replies_are_left_alone() {
        assert!(matches!(prepare("Plain **text** with a [link](https://example.com)."), Cow::Borrowed(_)));
        assert!(matches!(prepare("It costs $20 to $30."), Cow::Borrowed(_)));
        assert!(matches!(prepare(r"As shown \[1\]."), Cow::Borrowed(_)));
    }

    /// The typing cursor goes under a code block that's just closed, not on
    /// its fence (where it made the fence code: "```▌").
    #[test]
    fn the_cursor_keeps_off_a_closing_fence() {
        let cursor = |text: &str| {
            let mut text = text.to_string();
            push_cursor(&mut text);
            text
        };
        assert_eq!(cursor("Some words"), "Some words▌");
        assert_eq!(cursor("```python\nprint(1)\n```"), "```python\nprint(1)\n```\n\n▌");
        assert_eq!(cursor("- item\n  ```sh\n  ls\n  ```"), "- item\n  ```sh\n  ls\n  ```\n\n▌");
        assert_eq!(cursor("> ~~~\n> x\n> ~~~"), "> ~~~\n> x\n> ~~~\n\n▌");
        // Still inside the block: where the typing is.
        assert_eq!(cursor("```python\nprint(1)"), "```python\nprint(1)▌");
        assert_eq!(cursor("```python"), "```python▌");
        assert_eq!(cursor("````md\n```\nx\n```"), "````md\n```\nx\n```▌");
        assert_eq!(cursor("~~~\n```"), "~~~\n```▌");
        // A second block's opening fence isn't the first's closing one.
        assert_eq!(cursor("```\na\n```\n\n```"), "```\na\n```\n\n```▌");
        // After the fence's line it's under the block already.
        assert_eq!(cursor("```\na\n```\n"), "```\na\n```\n▌");
        // The block closed: the viewer sees a block and a cursor after it.
        let closed = cursor("```\na\n```");
        let events: Vec<Event> = Parser::new_ext(&closed, options(true)).collect();
        assert!(matches!(events.last(), Some(Event::End(TagEnd::Paragraph))), "{events:?}");
        assert!(events.iter().any(|e| matches!(e, Event::Text(t) if t.as_ref() == "▌")));
        assert!(events.iter().any(|e| matches!(e, Event::Text(t) if t.as_ref() == "a\n")));
    }
}
