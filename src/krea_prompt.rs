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
//! The brief rides along as a pinned priming turn (like role play's
//! preamble), and pictures are read with Detailed Vision (src/llm.rs tiles
//! the newest image) so small details make it into the prompt.

/// The priming turn pinned in front of the chat history.
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
Reply with ONLY the prompt paragraph: no title, no introduction, no quotes \
around it, no markdown, no lists, nothing after it.";

/// The model's scripted acknowledgement of the brief (keeps the user/model
/// turn alternation intact).
pub const ACK: &str = "Understood. I'll reply with only the Krea 2 prompt.";
