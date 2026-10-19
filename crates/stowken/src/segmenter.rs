//! Conversation → typed `Segment` decomposition.

use thiserror::Error;

use std::sync::Arc;

use crate::types::{
    Conversation, Message, MessageContent, Segment, SegmentType, Token, TokenizerAdapter,
};

/// Errors from segmentation.
#[derive(Debug, Error)]
pub enum SegmenterError {
    #[error(
        "message with role '{role}' contains text content but no tokenizer adapter is configured; \
         provide pre-tokenized content (Vec<u32>) or set a TokenizerAdapter on the vault"
    )]
    TextWithoutTokenizer { role: String },
    #[error("unknown message role: '{0}'")]
    UnknownRole(String),
}

/// Optional configuration for context-detection within user messages.
#[derive(Debug, Clone, Default)]
pub struct SegmenterConfig {
    /// Token sequence that marks the boundary between injected RAG context
    /// and the actual user query. When the delimiter is found inside a
    /// `UserTurn`, the message is split into a `Context` + `UserTurn` pair.
    pub context_delimiter: Option<Vec<Token>>,

    /// Maximum tokens per stored segment. When set, messages longer than
    /// this limit are split into a leading segment (original type) followed
    /// by one or more `Continuation` segments. For text messages, splits
    /// prefer paragraph boundaries (`\n\n`) before falling back to the hard
    /// token limit. For pre-tokenized messages, splits at the token limit.
    ///
    /// Smaller values create more, shorter segments with higher exact-dedup
    /// probability at the cost of more segment refs per conversation.
    pub max_segment_tokens: Option<usize>,
}

/// Decomposes a `Conversation` into a flat list of typed `Segment`s.
pub struct Segmenter {
    config: SegmenterConfig,
    tokenizer: Option<Arc<dyn TokenizerAdapter>>,
}

impl Segmenter {
    pub fn new(config: SegmenterConfig) -> Self {
        Self {
            config,
            tokenizer: None,
        }
    }

    pub fn with_tokenizer(mut self, adapter: Arc<dyn TokenizerAdapter>) -> Self {
        self.tokenizer = Some(adapter);
        self
    }

    /// Decompose a conversation into segments.
    pub fn segment(&self, conversation: &Conversation) -> Result<Vec<Segment>, SegmenterError> {
        let mut segments = Vec::new();
        for msg in &conversation.messages {
            let new_segs = self.segment_message(msg)?;
            segments.extend(new_segs);
        }
        Ok(segments)
    }

    fn segment_message(&self, msg: &Message) -> Result<Vec<Segment>, SegmenterError> {
        let seg_type = self.classify_role(msg)?;
        let meta = extract_metadata(msg);

        // For text content with a max_segment_tokens limit, split on paragraph
        // boundaries in text space before tokenizing so split points land at
        // natural sentence boundaries rather than mid-word.
        if let (MessageContent::Text(text), Some(max_tokens)) =
            (&msg.content, self.config.max_segment_tokens)
        {
            if let Some(adapter) = &self.tokenizer {
                let chunks = split_text_at_paragraphs(text, max_tokens, adapter.as_ref());
                if chunks.len() > 1 {
                    return Ok(chunks
                        .into_iter()
                        .enumerate()
                        .filter(|(_, tokens)| !tokens.is_empty())
                        .map(|(i, tokens)| Segment {
                            segment_type: if i == 0 {
                                seg_type.clone()
                            } else {
                                SegmentType::Continuation
                            },
                            tokens,
                            metadata: meta.clone(),
                        })
                        .collect());
                }
            }
        }

        let tokens = self.resolve_tokens(msg)?;

        // Context-delimiter split for user turns.
        if seg_type == SegmentType::UserTurn {
            if let Some(delimiter) = &self.config.context_delimiter {
                if let Some(split_pos) = find_subsequence(&tokens, delimiter) {
                    let context_tokens = tokens[..split_pos].to_vec();
                    let user_tokens = tokens[split_pos + delimiter.len()..].to_vec();
                    let mut result = Vec::new();
                    if !context_tokens.is_empty() {
                        result.push(Segment {
                            segment_type: SegmentType::Context,
                            tokens: context_tokens,
                            metadata: meta.clone(),
                        });
                    }
                    if !user_tokens.is_empty() {
                        result.push(Segment {
                            segment_type: SegmentType::UserTurn,
                            tokens: user_tokens,
                            metadata: meta,
                        });
                    }
                    return Ok(result);
                }
            }
        }

        // Token-only content (or text with no paragraph split needed): apply
        // max_segment_tokens hard split if configured.
        if let Some(max_tokens) = self.config.max_segment_tokens {
            if tokens.len() > max_tokens {
                return Ok(split_tokens_at_limit(&tokens, max_tokens, seg_type, meta));
            }
        }

        Ok(vec![Segment {
            segment_type: seg_type,
            tokens,
            metadata: meta,
        }])
    }

    fn classify_role(&self, msg: &Message) -> Result<SegmentType, SegmenterError> {
        match msg.role.as_str() {
            "system" => Ok(SegmentType::SystemPrompt),
            "user" => Ok(SegmentType::UserTurn),
            "assistant" => Ok(SegmentType::AssistantTurn),
            "tool" => {
                if msg.tool_call_id.is_some() {
                    Ok(SegmentType::ToolResult)
                } else {
                    Ok(SegmentType::ToolCall)
                }
            }
            other => Err(SegmenterError::UnknownRole(other.to_owned())),
        }
    }

    fn resolve_tokens(&self, msg: &Message) -> Result<Vec<Token>, SegmenterError> {
        match &msg.content {
            MessageContent::Tokens(t) => Ok(t.clone()),
            MessageContent::Text(text) => {
                if let Some(adapter) = &self.tokenizer {
                    Ok(adapter.tokenize(text))
                } else {
                    Err(SegmenterError::TextWithoutTokenizer {
                        role: msg.role.clone(),
                    })
                }
            }
        }
    }
}

/// Extract relevant fields from a message into a metadata map.
fn extract_metadata(msg: &Message) -> Option<std::collections::HashMap<String, serde_json::Value>> {
    let mut map = std::collections::HashMap::new();
    if let Some(name) = &msg.name {
        map.insert("name".to_owned(), serde_json::Value::String(name.clone()));
    }
    if let Some(id) = &msg.tool_call_id {
        map.insert(
            "tool_call_id".to_owned(),
            serde_json::Value::String(id.clone()),
        );
    }
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Split text on `\n\n` paragraph boundaries, tokenize each paragraph, then
/// merge adjacent paragraphs until adding the next would exceed `max_tokens`.
/// Falls back to a hard token-count split for paragraphs that are themselves
/// too long.
fn split_text_at_paragraphs(
    text: &str,
    max_tokens: usize,
    adapter: &dyn crate::types::TokenizerAdapter,
) -> Vec<Vec<Token>> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut chunks: Vec<Vec<Token>> = Vec::new();
    let mut current: Vec<Token> = Vec::new();

    for para in paragraphs {
        let para_tokens = adapter.tokenize(para);
        // A single paragraph that exceeds the limit is hard-split by tokens.
        if para_tokens.len() > max_tokens {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            for token_chunk in para_tokens.chunks(max_tokens) {
                chunks.push(token_chunk.to_vec());
            }
            continue;
        }
        // Append paragraph separator tokens (newline × 2) when joining.
        let sep_tokens = if current.is_empty() {
            vec![]
        } else {
            adapter.tokenize("\n\n")
        };
        if current.len() + sep_tokens.len() + para_tokens.len() > max_tokens && !current.is_empty()
        {
            chunks.push(std::mem::take(&mut current));
            current = para_tokens;
        } else {
            current.extend(sep_tokens);
            current.extend(para_tokens);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Split a pre-tokenized sequence into chunks of at most `max_tokens`, emitting
/// the first chunk with `seg_type` and subsequent chunks as `Continuation`.
fn split_tokens_at_limit(
    tokens: &[Token],
    max_tokens: usize,
    seg_type: SegmentType,
    meta: Option<std::collections::HashMap<String, serde_json::Value>>,
) -> Vec<Segment> {
    tokens
        .chunks(max_tokens)
        .enumerate()
        .map(|(i, chunk)| Segment {
            segment_type: if i == 0 {
                seg_type.clone()
            } else {
                SegmentType::Continuation
            },
            tokens: chunk.to_vec(),
            metadata: meta.clone(),
        })
        .collect()
}

/// Find the first occurrence of `needle` in `haystack`. Returns the start index.
fn find_subsequence(haystack: &[Token], needle: &[Token]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, MessageContent};

    fn make_msg(role: &str, tokens: Vec<Token>) -> Message {
        Message {
            role: role.to_owned(),
            content: MessageContent::Tokens(tokens),
            name: None,
            tool_call_id: None,
        }
    }

    fn make_tool_result(tokens: Vec<Token>) -> Message {
        Message {
            role: "tool".to_owned(),
            content: MessageContent::Tokens(tokens),
            name: None,
            tool_call_id: Some("call_abc".to_owned()),
        }
    }

    fn make_tool_call(tokens: Vec<Token>) -> Message {
        Message {
            role: "tool".to_owned(),
            content: MessageContent::Tokens(tokens),
            name: None,
            tool_call_id: None,
        }
    }

    fn make_conversation(messages: Vec<Message>) -> Conversation {
        Conversation {
            id: None,
            application: None,
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            messages,
            metadata: None,
        }
    }

    #[test]
    fn basic_role_mapping() {
        let segmenter = Segmenter::new(SegmenterConfig::default());
        let conv = make_conversation(vec![
            make_msg("system", vec![1, 2, 3]),
            make_msg("user", vec![4, 5, 6]),
            make_msg("assistant", vec![7, 8, 9]),
        ]);
        let segs = segmenter.segment(&conv).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].segment_type, SegmentType::SystemPrompt);
        assert_eq!(segs[1].segment_type, SegmentType::UserTurn);
        assert_eq!(segs[2].segment_type, SegmentType::AssistantTurn);
    }

    #[test]
    fn tool_call_vs_tool_result() {
        let segmenter = Segmenter::new(SegmenterConfig::default());
        let conv = make_conversation(vec![
            make_tool_call(vec![10, 11]),
            make_tool_result(vec![12, 13]),
        ]);
        let segs = segmenter.segment(&conv).unwrap();
        assert_eq!(segs[0].segment_type, SegmentType::ToolCall);
        assert_eq!(segs[1].segment_type, SegmentType::ToolResult);
    }

    #[test]
    fn context_delimiter_splits_user_turn() {
        let delimiter = vec![999u32, 998];
        let config = SegmenterConfig {
            context_delimiter: Some(delimiter.clone()),
            ..SegmenterConfig::default()
        };
        let segmenter = Segmenter::new(config);

        // Tokens: context [100, 200] ++ delimiter ++ user_query [300, 400]
        let tokens = vec![100u32, 200, 999, 998, 300, 400];
        let conv = make_conversation(vec![make_msg("user", tokens)]);
        let segs = segmenter.segment(&conv).unwrap();

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].segment_type, SegmentType::Context);
        assert_eq!(segs[0].tokens, vec![100u32, 200]);
        assert_eq!(segs[1].segment_type, SegmentType::UserTurn);
        assert_eq!(segs[1].tokens, vec![300u32, 400]);
    }

    #[test]
    fn no_delimiter_match_keeps_user_turn() {
        let delimiter = vec![999u32, 998];
        let config = SegmenterConfig {
            context_delimiter: Some(delimiter),
            ..SegmenterConfig::default()
        };
        let segmenter = Segmenter::new(config);
        let tokens = vec![100u32, 200, 300];
        let conv = make_conversation(vec![make_msg("user", tokens.clone())]);
        let segs = segmenter.segment(&conv).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].segment_type, SegmentType::UserTurn);
        assert_eq!(segs[0].tokens, tokens);
    }

    #[test]
    fn text_without_tokenizer_errors() {
        let segmenter = Segmenter::new(SegmenterConfig::default());
        let conv = make_conversation(vec![Message {
            role: "user".to_owned(),
            content: MessageContent::Text("hello".to_owned()),
            name: None,
            tool_call_id: None,
        }]);
        let result = segmenter.segment(&conv);
        assert!(matches!(
            result,
            Err(SegmenterError::TextWithoutTokenizer { .. })
        ));
    }

    #[test]
    fn unknown_role_errors() {
        let segmenter = Segmenter::new(SegmenterConfig::default());
        let conv = make_conversation(vec![make_msg("moderator", vec![1, 2, 3])]);
        let result = segmenter.segment(&conv);
        assert!(matches!(result, Err(SegmenterError::UnknownRole(_))));
    }

    #[test]
    fn tokens_preserved_exactly() {
        let segmenter = Segmenter::new(SegmenterConfig::default());
        let tokens = vec![0u32, 127, 128, 16_383, 16_384, u32::MAX];
        let conv = make_conversation(vec![make_msg("user", tokens.clone())]);
        let segs = segmenter.segment(&conv).unwrap();
        assert_eq!(segs[0].tokens, tokens);
    }
}
