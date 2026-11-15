//! PATCH(Clarity TagFlow): colouring for code blocks in the languages egui's
//! small built-in highlighter doesn't know — it has C/C++, Python, Rust and
//! TOML; replies are as often JavaScript, JSON, shell, SQL… A block in one
//! of those was plain text.
//!
//! The same kind of highlighter as egui's (comments, strings, keywords, the
//! rest), in the same colours, so a reply's blocks look alike whichever of
//! the two coloured them.

use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, FontId};

/// What a language's code is told apart by.
pub struct Language {
    keywords: &'static [&'static str],
    /// What starts a comment that runs to the end of its line.
    line_comments: &'static [&'static str],
    /// What opens and closes a comment that can run over lines.
    block_comment: Option<(&'static str, &'static str)>,
    /// What quotes a string.
    quotes: &'static [char],
    /// Keywords are in any case (SQL's `select` / `SELECT`).
    any_case: bool,
}

const C_LIKE: &[&str] = &["//"];
const HASH: &[&str] = &["#"];
const SLASH_STAR: Option<(&str, &str)> = Some(("/*", "*/"));
const BOTH_QUOTES: &[char] = &['"', '\''];

const JS: &[&str] = &[
    "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do",
    "else", "export", "extends", "false", "finally", "for", "from", "function", "get", "if", "import", "in",
    "instanceof", "let", "new", "null", "of", "return", "set", "static", "super", "switch", "this", "throw", "true",
    "try", "typeof", "undefined", "var", "void", "while", "with", "yield",
    // TypeScript's.
    "abstract", "any", "as", "boolean", "declare", "enum", "implements", "interface", "keyof", "namespace", "never",
    "number", "private", "protected", "public", "readonly", "string", "type", "unknown",
];
const JAVA: &[&str] = &[
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue",
    "default", "do", "double", "else", "enum", "extends", "false", "final", "finally", "float", "for", "goto", "if",
    "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "null", "package", "private",
    "protected", "public", "record", "return", "short", "static", "super", "switch", "synchronized", "this",
    "throw", "throws", "transient", "true", "try", "var", "void", "volatile", "while",
];
const KOTLIN: &[&str] = &[
    "abstract", "as", "break", "catch", "class", "companion", "continue", "data", "do", "else", "false", "finally",
    "for", "fun", "if", "import", "in", "interface", "internal", "is", "null", "object", "open", "override",
    "package", "private", "protected", "public", "return", "sealed", "super", "suspend", "this", "throw", "true",
    "try", "val", "var", "when", "while",
];
const GO: &[&str] = &[
    "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "false", "for", "func",
    "go", "goto", "if", "import", "interface", "map", "nil", "package", "range", "return", "select", "struct",
    "switch", "true", "type", "var",
];
const CSHARP: &[&str] = &[
    "abstract", "as", "async", "await", "base", "bool", "break", "byte", "case", "catch", "char", "class", "const",
    "continue", "decimal", "default", "delegate", "do", "double", "else", "enum", "event", "false", "finally",
    "float", "for", "foreach", "get", "goto", "if", "in", "int", "interface", "internal", "is", "lock", "long",
    "namespace", "new", "null", "object", "operator", "out", "override", "params", "private", "protected", "public",
    "readonly", "ref", "return", "sealed", "set", "short", "static", "string", "struct", "switch", "this", "throw",
    "true", "try", "typeof", "uint", "ulong", "using", "var", "virtual", "void", "while",
];
const SWIFT: &[&str] = &[
    "break", "case", "catch", "class", "continue", "default", "defer", "else", "enum", "extension", "false", "for",
    "func", "guard", "if", "import", "in", "init", "internal", "let", "nil", "override", "private", "protocol",
    "public", "repeat", "return", "self", "static", "struct", "super", "switch", "throw", "throws", "true", "try",
    "var", "while",
];
const PHP: &[&str] = &[
    "abstract", "array", "case", "catch", "class", "const", "default", "do", "echo", "else", "elseif", "extends",
    "false", "final", "finally", "for", "foreach", "function", "if", "implements", "interface", "namespace", "new",
    "null", "private", "protected", "public", "return", "static", "switch", "throw", "true", "try", "use", "var",
    "while",
];
const RUBY: &[&str] = &[
    "and", "begin", "case", "class", "def", "do", "else", "elsif", "end", "ensure", "false", "for", "if", "in",
    "include", "module", "nil", "not", "or", "raise", "require", "rescue", "return", "self", "then", "true",
    "unless", "until", "when", "while", "yield",
];
const LUA: &[&str] = &[
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto", "if", "in", "local", "nil",
    "not", "or", "repeat", "return", "then", "true", "until", "while",
];
const SHELL: &[&str] = &[
    "alias", "case", "cd", "declare", "do", "done", "echo", "elif", "else", "esac", "exit", "export", "fi", "for",
    "function", "if", "in", "local", "readonly", "return", "set", "source", "then", "unset", "until", "while",
];
const POWERSHELL: &[&str] = &[
    "begin", "catch", "do", "else", "elseif", "end", "finally", "for", "foreach", "function", "if", "in", "param",
    "process", "return", "switch", "throw", "try", "while",
];
const BATCH: &[&str] = &[
    "call", "do", "echo", "else", "endlocal", "exist", "exit", "for", "goto", "if", "in", "not", "rem", "set",
    "setlocal",
];
const SQL: &[&str] = &[
    "all", "alter", "and", "as", "asc", "avg", "begin", "between", "by", "case", "check", "commit", "constraint",
    "count", "create", "default", "delete", "desc", "distinct", "drop", "else", "end", "exists", "foreign", "from",
    "full", "group", "having", "in", "index", "inner", "insert", "into", "is", "join", "key", "left", "like",
    "limit", "max", "min", "not", "null", "offset", "on", "or", "order", "outer", "primary", "references", "right",
    "rollback", "select", "set", "sum", "table", "then", "union", "unique", "update", "values", "view", "when",
    "where",
];
const JSON: &[&str] = &["false", "null", "true"];
const YAML: &[&str] = &["false", "no", "null", "off", "on", "true", "yes"];
const DOCKER: &[&str] = &[
    "ADD", "ARG", "AS", "CMD", "COPY", "ENTRYPOINT", "ENV", "EXPOSE", "FROM", "LABEL", "RUN", "USER", "VOLUME",
    "WORKDIR",
];

const fn c_like(keywords: &'static [&'static str]) -> Language {
    Language { keywords, line_comments: C_LIKE, block_comment: SLASH_STAR, quotes: BOTH_QUOTES, any_case: false }
}

const fn hashed(keywords: &'static [&'static str]) -> Language {
    Language { keywords, line_comments: HASH, block_comment: None, quotes: BOTH_QUOTES, any_case: false }
}

/// The name a code block's first line gives its language, as it's looked up
/// by: its first word, in lower case ("Python title=x" is "python"), the
/// names egui's highlighter doesn't know put as ones it does.
pub fn language_name(info: &str) -> String {
    let name = info.split(|c: char| c.is_whitespace() || c == ',' || c == '{').next().unwrap_or("").to_ascii_lowercase();
    match name.as_str() {
        "cc" | "cxx" | "hxx" | "hh" => "cpp".to_string(),
        "python3" | "py3" | "pycon" => "py".to_string(),
        _ => name,
    }
}

/// The language called `name` (see [`language_name`]), if it's one of these.
pub fn language(name: &str) -> Option<Language> {
    Some(match name {
        "js" | "javascript" | "jsx" | "mjs" | "cjs" | "node" | "ts" | "typescript" | "tsx" => {
            Language { quotes: &['"', '\'', '`'], ..c_like(JS) }
        }
        "json" | "jsonc" | "json5" => Language { quotes: &['"'], ..c_like(JSON) },
        "java" => c_like(JAVA),
        "kotlin" | "kt" | "kts" => c_like(KOTLIN),
        "go" | "golang" => Language { quotes: &['"', '\'', '`'], ..c_like(GO) },
        "cs" | "csharp" | "c#" => c_like(CSHARP),
        "swift" => c_like(SWIFT),
        "php" => Language { line_comments: &["//", "#"], ..c_like(PHP) },
        "ruby" | "rb" => hashed(RUBY),
        "lua" => Language { line_comments: &["--"], block_comment: Some(("--[[", "]]")), ..hashed(LUA) },
        "sh" | "bash" | "zsh" | "shell" | "console" | "shellscript" => {
            Language { quotes: &['"', '\'', '`'], ..hashed(SHELL) }
        }
        "powershell" | "ps1" | "pwsh" => {
            Language { block_comment: Some(("<#", "#>")), any_case: true, ..hashed(POWERSHELL) }
        }
        "bat" | "batch" | "cmd" | "dos" => {
            Language { line_comments: &["::"], quotes: &['"'], any_case: true, ..hashed(BATCH) }
        }
        "sql" | "mysql" | "pgsql" | "postgres" | "postgresql" | "sqlite" | "plsql" | "tsql" => {
            Language { line_comments: &["--"], block_comment: SLASH_STAR, any_case: true, ..hashed(SQL) }
        }
        // (An apostrophe in a page's words isn't a quote.)
        "html" | "xml" | "svg" | "xhtml" | "vue" => {
            Language { line_comments: &[], block_comment: Some(("<!--", "-->")), quotes: &['"'], ..hashed(&[]) }
        }
        "css" | "scss" | "less" => Language { line_comments: &[], ..c_like(&[]) },
        "yaml" | "yml" => hashed(YAML),
        "ini" | "conf" | "cfg" | "properties" | "env" | "dotenv" => {
            Language { line_comments: &["#", ";"], quotes: &['"'], ..hashed(&[]) }
        }
        "dockerfile" | "docker" => hashed(DOCKER),
        _ => return None,
    })
}

/// The kinds of thing told apart (egui's own highlighter's).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Token {
    Comment,
    Keyword,
    Literal,
    Text,
    Punctuation,
    Whitespace,
}

impl Token {
    /// Its colour: egui's `CodeTheme`'s, dark and light.
    pub fn color(self, dark: bool) -> Color32 {
        match (self, dark) {
            (Token::Comment, true) => Color32::from_gray(120),
            (Token::Keyword, true) => Color32::from_rgb(255, 100, 100),
            (Token::Literal, true) => Color32::from_rgb(87, 165, 171),
            (Token::Text, true) => Color32::from_rgb(109, 147, 226),
            (Token::Punctuation, true) => Color32::LIGHT_GRAY,
            (Token::Comment, false) => Color32::GRAY,
            (Token::Keyword, false) => Color32::from_rgb(235, 0, 0),
            (Token::Literal, false) => Color32::from_rgb(153, 134, 255),
            (Token::Text, false) => Color32::from_rgb(37, 203, 105),
            (Token::Punctuation, false) => Color32::DARK_GRAY,
            (Token::Whitespace, _) => Color32::TRANSPARENT,
        }
    }
}

/// `code` cut into its tokens, end to end: each with how many bytes it is.
pub fn tokens(code: &str, language: &Language) -> Vec<(Token, usize)> {
    let mut out = Vec::new();
    let mut rest = code;
    // What came before: a comment mark only starts a comment at the start of
    // a line's words or after a break in them ("http://x" and "$#" aren't).
    let mut before: Option<char> = None;
    while let Some(first) = rest.chars().next() {
        let can_comment = before.is_none_or(|c| c.is_whitespace() || ";,{}()[]".contains(c));
        let block = language.block_comment.filter(|(open, _)| rest.starts_with(open));
        let (token, len) = if let Some((open, close)) = block {
            let end = rest[open.len()..].find(close).map_or(rest.len(), |at| open.len() + at + close.len());
            (Token::Comment, end)
        } else if can_comment && language.line_comments.iter().any(|mark| rest.starts_with(mark)) {
            (Token::Comment, rest.find('\n').unwrap_or(rest.len()))
        } else if language.quotes.contains(&first) {
            (Token::Text, string_len(rest, first))
        } else if first.is_alphanumeric() || first == '_' {
            let len = rest.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(rest.len());
            let word = &rest[..len];
            let keyword = if language.any_case {
                language.keywords.iter().any(|k| k.eq_ignore_ascii_case(word))
            } else {
                language.keywords.contains(&word)
            };
            (if keyword { Token::Keyword } else { Token::Literal }, len)
        } else if first.is_whitespace() {
            (Token::Whitespace, rest.find(|c: char| !c.is_whitespace()).unwrap_or(rest.len()))
        } else {
            (Token::Punctuation, first.len_utf8())
        };
        before = rest[..len].chars().next_back();
        out.push((token, len));
        rest = &rest[len..];
    }
    out
}

/// How long the string is that `code` starts with, in `quote`s: to its
/// closing quote (one after a backslash isn't it), or — unclosed — to the
/// end of its line. A backtick's can run over lines.
fn string_len(code: &str, quote: char) -> usize {
    let mut escaped = false;
    for (at, c) in code.char_indices().skip(1) {
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == quote {
            return at + c.len_utf8();
        } else if c == '\n' && quote != '`' {
            return at;
        }
    }
    code.len()
}

/// `code` in `language`'s colours (dark ones or light), set in `font`.
fn colored(code: &str, language: &Language, dark: bool, font: &FontId) -> LayoutJob {
    let mut job = LayoutJob::default();
    let mut at = 0;
    for (token, len) in tokens(code, language) {
        job.append(&code[at..at + len], 0.0, TextFormat::simple(font.clone(), token.color(dark)));
        at += len;
    }
    job
}

/// Kept from frame to frame: a block is coloured once, not each time it's
/// drawn.
#[derive(Default)]
struct Colorer;

impl egui::util::cache::ComputerMut<(&str, &str, bool, &FontId), Option<LayoutJob>> for Colorer {
    fn compute(&mut self, (code, name, dark, font): (&str, &str, bool, &FontId)) -> Option<LayoutJob> {
        language(name).map(|language| colored(code, &language, dark, font))
    }
}

type ColorCache = egui::util::cache::FrameCache<Option<LayoutJob>, Colorer>;

/// `code` coloured as the language called `name` (see [`language_name`]), or
/// `None` when it's not one of these (egui's highlighter has it then).
pub fn highlight(ui: &egui::Ui, code: &str, name: &str) -> Option<LayoutJob> {
    let dark = ui.visuals().dark_mode;
    let font = egui::TextStyle::Monospace.resolve(ui.style());
    ui.ctx().memory_mut(|memory| memory.caches.cache::<ColorCache>().get((code, name, dark, &font)).clone())
}
