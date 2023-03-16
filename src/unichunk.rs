use unicode_segmentation::UnicodeSegmentation;

pub fn split_once<'a>(s: &'a str, limit: usize) -> (std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>) {
    if s.len() <= limit {
        return (std::borrow::Cow::Borrowed(s), std::borrow::Cow::Borrowed(""));
    }

    let breakpoints = unicode_linebreak::linebreaks(&s).collect::<Vec<_>>();

    // Try to break on a mandatory line break location first.
    for &(i, opportunity) in breakpoints.iter().rev() {
        if opportunity != unicode_linebreak::BreakOpportunity::Mandatory {
            continue;
        }
        if i <= limit && i > 0 {
            let (head, tail) = s.split_at(i);
            return (std::borrow::Cow::Borrowed(head), std::borrow::Cow::Borrowed(tail));
        }
    }

    // Break on sentences if we can't break cleanly.
    for (i, _) in s.split_sentence_bound_indices().collect::<Vec<_>>().into_iter().rev() {
        if i <= limit && i > 0 {
            let (head, tail) = s.split_at(i);
            return (std::borrow::Cow::Borrowed(head), std::borrow::Cow::Borrowed(tail));
        }
    }

    // Then, try to break on an allowed line break location. This might be a space in the middle of a sentence.
    for &(i, opportunity) in breakpoints.iter().rev() {
        if opportunity != unicode_linebreak::BreakOpportunity::Allowed {
            continue;
        }
        if i <= limit && i > 0 {
            let (head, tail) = s.split_at(i);
            return (std::borrow::Cow::Borrowed(head), std::borrow::Cow::Borrowed(tail));
        }
    }

    // Failing that, break between graphemes instead.
    for (i, _) in s.grapheme_indices(true).rev() {
        if i <= limit && i > 0 {
            let (head, tail) = s.split_at(i);
            return (std::borrow::Cow::Borrowed(head), std::borrow::Cow::Borrowed(tail));
        }
    }

    // Break on Unicode codepoint if we can't break on a grapheme index. This can split 👨‍👩‍👦 into 👨 and 👨‍👩.
    for (i, _) in s.char_indices().rev() {
        if i <= limit && i > 0 {
            let (head, tail) = s.split_at(i);
            return (std::borrow::Cow::Borrowed(head), std::borrow::Cow::Borrowed(tail));
        }
    }

    // Just kind of screwed, split at a byte position.
    let (head, tail) = s.as_bytes().split_at(limit);
    (String::from_utf8_lossy(head), String::from_utf8_lossy(tail))
}

pub struct Chunker {
    buf: String,
    limit: usize,
}

impl Chunker {
    pub fn new(limit: usize) -> Self {
        Self { buf: String::new(), limit }
    }

    pub fn push(&mut self, s: &str) -> Vec<String> {
        let mut chunks = vec![];

        self.buf.push_str(s);
        loop {
            let (head, tail) = split_once(&self.buf, self.limit);
            if tail.is_empty() {
                break;
            }
            chunks.push(head.to_string());
            self.buf = tail.to_string();
        }
        chunks
    }

    pub fn flush(self) -> String {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_empty() {
        let (head, tail) = split_once("", 7);
        assert_eq!(head, "");
        assert_eq!(tail, "");
    }

    #[test]
    fn test_split_once_easy() {
        let (head, tail) = split_once("hello world", 7);
        assert_eq!(head, "hello ");
        assert_eq!(tail, "world");
    }

    #[test]
    fn test_split_once_precise() {
        let (head, tail) = split_once("a a a b b b c c", 4);
        assert_eq!(head, "a a ");
        assert_eq!(tail, "a b b b c c");
    }

    #[test]
    fn test_split_once_break_word() {
        let (head, tail) = split_once("aaaaabb", 2);
        assert_eq!(head, "aa");
        assert_eq!(tail, "aaabb");
    }

    #[test]
    fn test_split_once_break_linebreak_mandatory() {
        let (head, tail) = split_once("aa\naa abb", 4);
        assert_eq!(head, "aa\n");
        assert_eq!(tail, "aa abb");
    }

    #[test]
    fn test_split_once_break_sentence() {
        let (head, tail) = split_once("A a. A a [...] abb.", 7);
        assert_eq!(head, "A a. ");
        assert_eq!(tail, "A a [...] abb.");
    }

    #[test]
    fn test_split_once_break_no_family_separation() {
        let (head, tail) = split_once("hello 👨‍👩‍👦 world", 8);
        assert_eq!(head, "hello ");
        assert_eq!(tail, "👨‍👩‍👦 world");
    }

    #[test]
    fn test_split_once_break_family_separation() {
        let (head, tail) = split_once("👨‍👩‍👦", 4);
        assert_eq!(head, "👨");
        assert_eq!(tail, "\u{200d}👩\u{200d}👦");
    }

    #[test]
    fn test_split_once_break_desperate() {
        let (head, tail) = split_once("👨‍👩‍👦", 2);
        assert_eq!(head, "�");
        assert_eq!(tail, "��\u{200d}👩\u{200d}👦");
    }
}
