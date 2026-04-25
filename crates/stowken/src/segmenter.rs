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
        let tokens = self.resolve_tokens(msg)?;
        let seg_type = self.classify_role(msg)?;

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
                            metadata: extract_metadata(msg),
                        });
                    }
                    if !user_tokens.is_empty() {
                        result.push(Segment {
                            segment_type: SegmentType::UserTurn,
                            tokens: user_tokens,
                            metadata: extract_metadata(msg),
                        });
                    }
                    return Ok(result);
                }
            }
        }

        Ok(vec![Segment {
            segment_type: seg_type,
            tokens,
            metadata: extract_metadata(msg),
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
        map.insert("tool_call_id".to_owned(), serde_json::Value::String(id.clone()));
    }
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Find the first occurrence of `needle` in `haystack`. Returns the start index.
fn find_subsequence(haystack: &[Token], needle: &[Token]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
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
        let config = SegmenterConfig { context_delimiter: Some(delimiter.clone()) };
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
        let config = SegmenterConfig { context_delimiter: Some(delimiter) };
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
        assert!(matches!(result, Err(SegmenterError::TextWithoutTokenizer { .. })));
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
