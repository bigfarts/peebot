use unicode_segmentation::UnicodeSegmentation;

pub fn split_once(s: &str, limit: usize) -> (&str, &str) {
    if s.len() <= limit {
        return (s, "");
    }

    let breakpoints = unicode_linebreak::linebreaks(&s).collect::<Vec<_>>();

    // Try to break on a mandatory line break location first.
    for &(i, opportunity) in breakpoints.iter().rev() {
        if opportunity != unicode_linebreak::BreakOpportunity::Mandatory {
            continue;
        }
        if i <= limit {
            return s.split_at(i);
        }
    }

    // Then, try to break on an allowed line break location.
    for &(i, opportunity) in breakpoints.iter().rev() {
        if opportunity != unicode_linebreak::BreakOpportunity::Allowed {
            continue;
        }
        if i <= limit {
            return s.split_at(i);
        }
    }

    // Failing that, break on a grapheme index instead.
    for (i, _) in s.grapheme_indices(true).rev() {
        if i <= limit {
            return s.split_at(i);
        }
    }

    // Just kind of screwed, split at a byte position.
    s.split_at(limit)
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
        assert_eq!(split_once("", 7), ("", ""));
    }

    #[test]
    fn test_split_once_easy() {
        assert_eq!(split_once("hello world", 7), ("hello ", "world"));
    }

    #[test]
    fn test_split_once_precise() {
        assert_eq!(split_once("a a a b b b c c", 4), ("a a ", "a b b b c c"));
    }

    #[test]
    fn test_split_once_break_word() {
        assert_eq!(split_once("aaaaabb", 2), ("aa", "aaabb"));
    }

    #[test]
    fn test_split_once_break_linebreak_mandatory() {
        assert_eq!(split_once("aa\naa abb", 7), ("aa\n", "aa abb"));
    }

    #[test]
    fn test_split_once_break_no_family_separation() {
        assert_eq!(split_once("hello 👨‍👩‍👦 world", 8), ("hello ", "👨‍👩‍👦 world"));
    }
}
