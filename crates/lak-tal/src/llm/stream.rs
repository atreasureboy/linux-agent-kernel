//! Shared stream utilities for line-oriented protocols (SSE / NDJSON).
//!
//! Network chunk boundaries never align with line boundaries. A naive
//! per-chunk parser silently drops every JSON line that is split across
//! two chunks. [`LineStream`] buffers partial data and yields one item
//! per complete line, so parsers can treat every line as a whole.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;

/// Adapter that turns a byte-chunk stream into a stream of complete lines.
///
/// Lines are split on `\n`; trailing `\r` is stripped. Any trailing
/// fragment without a terminating newline is held back until more data
/// arrives (or the stream ends, in which case it is flushed).
pub struct LineStream<S> {
    inner: S,
    buffer: Vec<u8>,
    lines: VecDeque<Vec<u8>>,
    finished: bool,
}

impl<S> LineStream<S> {
    /// Wrap a byte-chunk stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
            lines: VecDeque::new(),
            finished: false,
        }
    }

    /// Move all complete lines out of the internal buffer.
    fn extract_lines(&mut self) {
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = self.buffer.drain(..=pos).collect();
            let mut line = raw;
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if !line.is_empty() {
                self.lines.push_back(line);
            }
        }
    }
}

impl<S, E> Stream for LineStream<S>
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Unpin,
{
    /// One complete line (lossy UTF-8 decode).
    type Item = Result<String, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            if let Some(line) = this.lines.pop_front() {
                let text = String::from_utf8_lossy(&line).into_owned();
                return Poll::Ready(Some(Ok(text)));
            }

            if this.finished {
                if !this.buffer.is_empty() {
                    let rest = std::mem::take(&mut this.buffer);
                    let text = String::from_utf8_lossy(&rest).into_owned();
                    let text = text.trim_end_matches(['\r', '\n']).to_string();
                    if !text.is_empty() {
                        return Poll::Ready(Some(Ok(text)));
                    }
                }
                return Poll::Ready(None);
            }

            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    this.buffer.extend_from_slice(&chunk);
                    this.extract_lines();
                }
                Poll::Ready(Some(Err(e))) => {
                    this.finished = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    this.finished = true;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn chunk_stream(
        chunks: Vec<&str>,
    ) -> LineStream<futures::stream::Iter<std::vec::IntoIter<Result<bytes::Bytes, std::io::Error>>>>
    {
        let items: Vec<Result<bytes::Bytes, std::io::Error>> = chunks
            .into_iter()
            .map(|c| Ok(bytes::Bytes::from(c.to_string())))
            .collect();
        LineStream::new(futures::stream::iter(items))
    }

    #[tokio::test]
    async fn test_split_line_is_reassembled() {
        // One JSON line split across three network chunks
        let mut stream = chunk_stream(vec![r#"{"a":""#, "he", "llo\"}\nnext\n"]);
        let mut out = Vec::new();
        while let Some(line) = stream.next().await {
            out.push(line.unwrap());
        }
        assert_eq!(out, vec![r#"{"a":"hello"}"#, "next"]);
    }

    #[tokio::test]
    async fn test_multiple_lines_per_chunk() {
        let mut stream = chunk_stream(vec!["l1\nl2\nl3\n"]);
        let mut out = Vec::new();
        while let Some(line) = stream.next().await {
            out.push(line.unwrap());
        }
        assert_eq!(out, vec!["l1", "l2", "l3"]);
    }

    #[tokio::test]
    async fn test_trailing_fragment_is_flushed() {
        let mut stream = chunk_stream(vec!["a\nb"]);
        let mut out = Vec::new();
        while let Some(line) = stream.next().await {
            out.push(line.unwrap());
        }
        assert_eq!(out, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn test_crlf_is_stripped() {
        let mut stream = chunk_stream(vec!["line\r\n"]);
        assert_eq!(stream.next().await.unwrap().unwrap(), "line");
    }
}
