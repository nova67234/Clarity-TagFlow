//! The AI's clock: the message being answered goes to the AI with the date
//! and time on this PC in front of it, so it knows when "now" is — what day
//! it is, how long until Friday, whether it's morning. The chat keeps the
//! message as typed; only what the AI reads has the line. The time is read
//! on the PC being chatted on, so a Link host answering for it still goes by
//! its user's clock.

use chrono::{DateTime, TimeZone};

use super::CmdMsg;

/// The line for `now`: "[The user's local date and time: Saturday,
/// 31 October 2026, 11:42 PM (UTC+01:00)]".
pub(super) fn line<Tz: TimeZone>(now: &DateTime<Tz>) -> String
where
    Tz::Offset: std::fmt::Display,
{
    format!(
        "[The user's local date and time: {} (UTC{})]",
        now.format("%A, %-d %B %Y, %-I:%M %p"),
        now.format("%:z")
    )
}

/// Put `line` in front of the newest user message in `msgs` (the one being
/// answered; pinned briefs come first, so they're never it).
pub(super) fn stamp(msgs: &mut [CmdMsg], line: &str) {
    if let Some(m) = msgs.iter_mut().rev().find(|m| m.user) {
        m.text = format!("{line}\n\n{}", m.text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    fn msg(user: bool, text: &str) -> CmdMsg {
        CmdMsg { user, text: text.to_string(), image: None }
    }

    #[test]
    fn the_line_says_the_day_date_time_and_offset() {
        let now = FixedOffset::east_opt(3600).unwrap().with_ymd_and_hms(2026, 10, 31, 23, 42, 0).unwrap();
        assert_eq!(line(&now), "[The user's local date and time: Saturday, 31 October 2026, 11:42 PM (UTC+01:00)]");
        let now = FixedOffset::west_opt(5 * 3600).unwrap().with_ymd_and_hms(2026, 1, 5, 9, 3, 0).unwrap();
        assert_eq!(line(&now), "[The user's local date and time: Monday, 5 January 2026, 9:03 AM (UTC-05:00)]");
    }

    #[test]
    fn only_the_newest_user_message_is_stamped() {
        // A pinned brief and its reply, an earlier turn, then the message.
        let mut msgs =
            vec![msg(true, "brief"), msg(false, "ok"), msg(true, "hi"), msg(false, "hello"), msg(true, "what day is it?")];
        stamp(&mut msgs, "[clock]");
        let texts: Vec<&str> = msgs.iter().map(|m| m.text.as_str()).collect();
        assert_eq!(texts, ["brief", "ok", "hi", "hello", "[clock]\n\nwhat day is it?"]);
        // Nothing to stamp: nothing happens.
        let mut none = vec![msg(false, "hello")];
        stamp(&mut none, "[clock]");
        assert_eq!(none[0].text, "hello");
    }
}
