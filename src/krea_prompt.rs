//! Krea 2 prompt writer for the AI Chat (toggled from the input card's tools
//! menu): turns an attached picture — or an idea typed in words — into a
//! prompt for Krea 2, the text-to-image model the Generate view runs.
//!
//! The brief encodes how Krea 2 reads prompts. Its text encoder is a language
//! model (Qwen3-VL), so it wants natural descriptive prose rather than tag
//! lists — ComfyUI's krea2 encoder even conditions on a system line asking
//! for "the color, shape, size, texture, quantity, text, spatial
//! relationships of the objects and background", which the brief mirrors; concrete physical detail beats adjectives; text to render goes in
//! double quotes; the medium should be named outright; and Krea 2 Turbo runs
//! at CFG 1, so negative phrasing has no effect — everything is said
//! positively. Run locally, Krea 2 uses the prompt as-is (no hosted prompt
//! expander), so the prompt has to carry all the detail itself.
//!
//! Anything that isn't a picture, an idea or a change ("hi", "thanks", a
//! question) gets an ordinary short answer instead: the brief used to ask
//! for nothing but prompts, so a greeting got no reply at all.
//!
//! The brief rides along as a pinned priming turn (like role play's
//! preamble), and a picture is read with Detailed Vision when it's sent
//! (src/llm.rs: the whole picture and close-up sections of it) so small
//! details make it into the prompt. A change asked for afterwards ("make it
//! night") gets the picture whole: the prompt already written carries the
//! details, and reading every section again made each change as slow to
//! start as the first prompt.
//!
//! A model that thinks first is told to keep that to notes ([`brief`]): left
//! to itself it writes the whole prompt in its head and then again as the
//! answer — twice the wait for the same words.
//!
//! A reply that is a prompt ([`is_prompt`]) can be taken to Text to Image
//! from the chat.

/// The priming turn pinned in front of the chat history, for a model that
/// answers straight away (see [`brief`]).
pub const BRIEF: &str = "[Krea 2 prompt writer]\n\
You write prompts for Krea 2, a text-to-image model. Its text encoder is a \
language model, so it understands natural, descriptive English prose — never \
write tag lists.\n\
\n\
- When I send a picture: write the Krea 2 prompt that would recreate that \
picture as faithfully as possible. You may also get zoomed-in sections of the \
same picture; use them for small details only.\n\
- When I describe an idea in words: turn it into a Krea 2 prompt.\n\
- When I ask for a change (\"make it night\", \"more cinematic\", \"as an oil \
painting\"): rewrite the WHOLE prompt with the change applied.\n\
- When I'm just talking to you — a greeting, thanks, a question about you, \
about Krea 2 or about a prompt you wrote — answer me normally, warmly and in \
a few sentences, in the language I write in. Don't write a prompt then. If \
I haven't asked for anything yet, say you can turn a picture or an idea \
into a Krea 2 prompt. Even a few words that describe something to picture \
(\"a cat in the rain\") are an idea: write the prompt.\n\
\n\
How to write the prompt:\n\
- One flowing paragraph of 4 to 8 sentences (roughly 90 to 200 words), the \
most important things first.\n\
- Order: the main subject and what it is doing; then its appearance — \
clothing, materials, colours, textures, hair, expression; then pose, framing \
and camera angle; then the lighting (direction, quality, colour); then the \
setting and background; last, the overall look — the medium (photograph, \
oil painting, watercolor, flat vector illustration, anime illustration, 3D \
render, …), the visual style and the colour palette.\n\
- Be concrete and physical: exact colours, shapes and sizes, fabrics, \
materials and textures, how many people or objects there are, and where \
things sit in the frame and relative to each other (left, right, in front \
of, behind, foreground, background).\n\
- Include the small details that make the picture specific — they are easy \
to miss, so look for them: jewelry and accessories (earrings, necklaces, \
rings, glasses, hair clips, scarves, scrunchies, bows), nail and toenail \
polish colour, makeup, tattoos, eye colour, fabric patterns and prints, \
logos, small objects being held, and surface details such as moss, lichen, \
rust or water droplets.\n\
- Always name the medium. For a photograph, give the shot type, the lens feel \
and the depth of field (for example \"shot on a 50mm lens with a shallow \
depth of field\").\n\
- Visible text or lettering: write the words inside double quotes exactly as \
they appear, and say where and in what style they are written.\n\
- Describe only what should be in the image, phrased positively. Never use \
negative phrasing such as \"no blur\" or \"without people\" — Krea 2 ignores \
it.\n\
- No quality tags or filler: never write masterpiece, best quality, 8k, 4k, \
ultra detailed, highly detailed, hyperrealistic, trending, award-winning.\n\
- Only describe what you can actually see (or what I asked for). Don't invent \
names, brands or a backstory.\n\
\n\
When you write a prompt, reply with ONLY the prompt paragraph: no title, no \
introduction, no quotes around it, no markdown, no lists, nothing after it.";

/// The model's scripted acknowledgement of the brief (keeps the user/model
/// turn alternation intact).
pub const ACK: &str = "Understood. When you send a picture, an idea or a change, I'll reply with only the \
                        Krea 2 prompt; otherwise I'll just talk with you.";

/// The brief for the model that answers: for one that thinks first (`thinks`
/// — the model does, and thinking isn't Off) with a rule about the thought,
/// as role play has. At High and Max (`deep`) the user asked for more
/// thought, so it may think the picture through — as notes all the same.
pub fn brief(thinks: bool, deep: bool) -> String {
    if !thinks {
        return BRIEF.to_string();
    }
    let rule = if deep {
        "- When you reason privately before answering, look the picture or the idea over thoroughly — \
         but as notes of what you see and what to include, never as drafts of the prompt: the prompt \
         is written once, in your answer."
    } else {
        "- Think briefly: when you reason privately before answering, a few short notes of what you \
         see and what to include are enough — decide, then write. Never draft or revise the prompt \
         in your head: it is written once, in your answer."
    };
    format!("{BRIEF}\n\nBefore you answer:\n{rule}")
}

/// A prompt is this many words at least (the brief asks for 90 to 200; what
/// the model says when it's just talking is "a few sentences").
const PROMPT_WORDS: usize = 60;
/// With a picture sent, where a prompt is what's asked for.
const PICTURE_PROMPT_WORDS: usize = 30;

/// A finished reply, written with the brief, is a prompt rather than talk:
/// one paragraph of a prompt's length (`picture`: the message it answers
/// came with a picture).
pub fn is_prompt(reply: &str, picture: bool) -> bool {
    let reply = reply.trim();
    let one_paragraph = !reply.lines().any(|line| line.trim().is_empty());
    let words = reply.split_whitespace().count();
    one_paragraph && words >= if picture { PICTURE_PROMPT_WORDS } else { PROMPT_WORDS }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that thinks is told to keep the thought to notes; one that
    /// answers straight away gets the brief as it is.
    #[test]
    fn a_thinking_model_is_told_not_to_draft() {
        assert_eq!(brief(false, false), BRIEF);
        assert_eq!(brief(false, true), BRIEF);
        let (quick, deep) = (brief(true, false), brief(true, true));
        for b in [&quick, &deep] {
            assert!(b.starts_with(BRIEF) && b.len() > BRIEF.len());
            assert!(b.contains("written once, in your answer"), "{b}");
        }
        assert!(quick.contains("Think briefly") && !deep.contains("Think briefly"));
        assert!(deep.contains("thoroughly"));
    }

    /// A prompt is one paragraph of some length; a greeting, thanks or an
    /// answer in a few sentences isn't.
    #[test]
    fn a_prompt_is_told_from_talk() {
        let prompt = "A weathered lighthouse keeper in a navy wool sweater sits at a scarred oak table, \
                      reading a creased letter by the amber glow of a brass oil lamp. Rain streaks the \
                      small window behind him, where a storm-grey sea heaves under a dark sky. His \
                      silver beard catches the warm light, and deep lines frame his pale blue eyes. A \
                      chipped enamel mug steams beside a coil of rope. Shot on a 50mm lens with a \
                      shallow depth of field, a moody photograph in muted teal and amber.";
        assert!(is_prompt(prompt, false) && is_prompt(prompt, true));
        assert!(is_prompt(&format!("  {prompt}\n"), false));
        for talk in [
            "",
            "Hi! Send me a picture or describe an idea and I'll turn it into a Krea 2 prompt.",
            "You're welcome! Want me to change anything about it?",
        ] {
            assert!(!is_prompt(talk, false) && !is_prompt(talk, true), "{talk}");
        }
        // A short prompt for a picture still is one; the same words with no
        // picture read as talk.
        let short = "A grey tabby cat curled up asleep on a sunlit wooden windowsill beside a potted \
                     fern, soft morning light, shallow depth of field, a cosy photograph in warm \
                     honey tones with gentle shadows.";
        assert!(is_prompt(short, true) && !is_prompt(short, false));
        // An answer set out in parts isn't a prompt, however long.
        let parts = format!("Here is what I changed:\n\n{prompt}");
        assert!(!is_prompt(&parts, false) && !is_prompt(&parts, true));
    }
}
