use crate::module::{Flag, Module, ModuleVisitor, Param, ParamId};
use crate::tensor::{Bool, Int, Tensor, backend::Backend};
use alloc::{string::String, vec, vec::Vec};

/// Errors constructing a parameter group.
#[derive(Debug)]
pub enum ParamGroupError {
    /// An invalid regular expression.
    InvalidPatternError(String),
}

/// Select module-owned parameters and flags by identity or path.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ParamGroup {
    matcher: ParamGroupMatcher,
    excludes: Option<ParamGroupMatcher>,
}

impl ParamGroup {
    /// Test an id and optional module path.
    pub fn matches(&self, id: &ParamId, path: Option<&str>) -> bool {
        self.matcher.matches(id, path)
            && !self.excludes.as_ref().is_some_and(|m| m.matches(id, path))
    }

    /// Collect all parameter ids, including runtime flags, in a module subtree.
    pub fn ids_from_module<B: Backend, M: Module<B>>(module: M) -> Self {
        #[derive(Default)]
        struct Collector(Vec<ParamId>);
        impl<B: Backend> ModuleVisitor<B> for Collector {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
                self.0.push(p.id);
            }
            fn visit_int<const D: usize>(&mut self, p: &Param<Tensor<B, D, Int>>) {
                self.0.push(p.id);
            }
            fn visit_bool<const D: usize>(&mut self, p: &Param<Tensor<B, D, Bool>>) {
                self.0.push(p.id);
            }
            fn visit_flag(&mut self, p: &Param<Flag>) {
                self.0.push(p.id);
            }
        }
        let mut collector = Collector::default();
        module.visit(&mut collector);
        Self::from_ids(collector.0)
    }

    /// Match one exact path.
    pub fn from_path(path: impl Into<String>) -> Self {
        Self::from_paths(vec![path])
    }
    /// Match any of the exact paths.
    pub fn from_paths(paths: Vec<impl Into<String>>) -> Self {
        Self::path(PathMatcher::Exact(
            paths.into_iter().map(Into::into).collect(),
        ))
    }
    /// Match a path containing the substring.
    pub fn from_predicate(path: impl Into<String>) -> Self {
        Self::from_predicates(vec![path])
    }
    /// Match a path containing every substring.
    pub fn from_predicates(paths: Vec<impl Into<String>>) -> Self {
        Self::path(PathMatcher::Include(
            paths.into_iter().map(Into::into).collect(),
        ))
    }
    /// Match a path containing any substring. An empty list matches every supplied path.
    pub fn from_any_predicates(paths: Vec<impl Into<String>>) -> Self {
        Self::any(paths.into_iter().map(Self::from_predicate).collect())
    }

    #[cfg(all(feature = "std", target_has_atomic = "ptr"))]
    /// Match one regular expression.
    pub fn from_regex(pattern: impl AsRef<str>) -> Result<Self, ParamGroupError> {
        Self::from_regexes(vec![pattern])
    }
    #[cfg(all(feature = "std", target_has_atomic = "ptr"))]
    /// Match every regular expression.
    pub fn from_regexes(patterns: Vec<impl AsRef<str>>) -> Result<Self, ParamGroupError> {
        let patterns = patterns
            .into_iter()
            .map(|p| {
                regex::Regex::new(p.as_ref()).map_err(|e| {
                    ParamGroupError::InvalidPatternError(alloc::format!(
                        "Invalid regex pattern: {e}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::path(PathMatcher::Regex(patterns)))
    }
    #[cfg(all(feature = "std", target_has_atomic = "ptr"))]
    /// Match any regular expression. An empty list matches every supplied path.
    pub fn from_any_regexes(patterns: Vec<impl AsRef<str>>) -> Result<Self, ParamGroupError> {
        Ok(Self::any(
            patterns
                .into_iter()
                .map(Self::from_regex)
                .collect::<Result<_, _>>()?,
        ))
    }

    /// Match every parameter and flag.
    pub fn all() -> Self {
        Self {
            matcher: ParamGroupMatcher::All,
            excludes: None,
        }
    }
    /// Match explicit ids.
    pub fn from_ids(ids: Vec<ParamId>) -> Self {
        Self {
            matcher: ParamGroupMatcher::Explicit(ids),
            excludes: None,
        }
    }
    /// Union the inclusion matchers, without retaining either group's exclusions.
    pub fn fuse(self, other: &Self) -> Self {
        Self {
            matcher: self.matcher.fuse(other.matcher.clone()),
            excludes: None,
        }
    }
    /// Exclude the other group's inclusion matcher.
    pub fn exclude(mut self, group: Self) -> Self {
        self.excludes = Some(match self.excludes {
            Some(matcher) => matcher.fuse(group.matcher),
            None => group.matcher,
        });
        self
    }
    fn path(matcher: PathMatcher) -> Self {
        Self {
            matcher: ParamGroupMatcher::Path(matcher),
            excludes: None,
        }
    }
    fn any(mut groups: Vec<Self>) -> Self {
        let Some(first) = groups.pop() else {
            return Self::path(PathMatcher::Include(vec![]));
        };
        groups
            .into_iter()
            .fold(first, |group, other| group.fuse(&other))
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ParamGroupMatcher {
    All,
    #[serde(with = "ids_serde")]
    Explicit(Vec<ParamId>),
    Path(PathMatcher),
    Combined(Vec<Self>),
}

mod ids_serde {
    use super::*;
    use serde::{Deserialize, Serialize};
    pub fn serialize<S: serde::Serializer>(ids: &[ParamId], s: S) -> Result<S::Ok, S::Error> {
        ids.iter()
            .map(|id| id.serialize())
            .collect::<Vec<_>>()
            .serialize(s)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<ParamId>, D::Error> {
        Vec::<String>::deserialize(d)?
            .into_iter()
            .map(|id| {
                ParamId::try_deserialize(&id)
                    .ok_or_else(|| serde::de::Error::custom("Invalid parameter id"))
            })
            .collect()
    }
}

impl ParamGroupMatcher {
    fn matches(&self, id: &ParamId, path: Option<&str>) -> bool {
        match self {
            Self::All => true,
            Self::Explicit(ids) => ids.contains(id),
            Self::Path(matcher) => path.is_some_and(|p| matcher.matches(p)),
            Self::Combined(matchers) => matchers.iter().any(|m| m.matches(id, path)),
        }
    }
    fn fuse(self, other: Self) -> Self {
        match (self, other) {
            (Self::All, _) | (_, Self::All) => Self::All,
            (Self::Combined(mut a), Self::Combined(b)) => {
                a.extend(b);
                Self::Combined(a)
            }
            (Self::Combined(mut a), b) | (b, Self::Combined(mut a)) => {
                a.push(b);
                Self::Combined(a)
            }
            (a, b) => Self::Combined(vec![a, b]),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PathMatcher {
    Exact(Vec<String>),
    #[cfg(all(feature = "std", target_has_atomic = "ptr"))]
    #[serde(with = "regex_serde")]
    Regex(Vec<regex::Regex>),
    Include(Vec<String>),
}

impl PathMatcher {
    fn matches(&self, path: &str) -> bool {
        match self {
            Self::Exact(paths) => paths.iter().any(|p| p == path),
            Self::Include(parts) => parts.iter().all(|p| path.contains(p)),
            #[cfg(all(feature = "std", target_has_atomic = "ptr"))]
            Self::Regex(patterns) => patterns.iter().all(|p| p.is_match(path)),
        }
    }
}

#[cfg(all(feature = "std", target_has_atomic = "ptr"))]
mod regex_serde {
    use super::*;
    use serde::{Deserialize, Serialize};
    pub fn serialize<S: serde::Serializer>(
        patterns: &[regex::Regex],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        patterns
            .iter()
            .map(|p| p.as_str())
            .collect::<Vec<_>>()
            .serialize(s)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<regex::Regex>, D::Error> {
        Vec::<String>::deserialize(d)?
            .into_iter()
            .map(|p| regex::Regex::new(&p).map_err(serde::de::Error::custom))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_serialization_preserves_ids_paths_and_exclusions() {
        let id = ParamId::from(17);
        let group = ParamGroup::from_ids(vec![id])
            .fuse(&ParamGroup::from_predicate("training"))
            .exclude(ParamGroup::from_path("skip.training"));
        let bytes = serde_json::to_vec(&group).unwrap();
        let restored: ParamGroup = serde_json::from_slice(&bytes).unwrap();
        assert!(restored.matches(&id, None));
        assert!(restored.matches(&ParamId::from(18), Some("dropout.training")));
        assert!(!restored.matches(&id, Some("skip.training")));
        assert!(!restored.matches(&ParamId::from(18), None));
    }

    #[cfg(all(feature = "std", target_has_atomic = "ptr"))]
    #[test]
    fn group_regex_serialization_preserves_matchers() {
        let group = ParamGroup::from_regex(r"^layers\.\d+\.training$").unwrap();
        let bytes = serde_json::to_vec(&group).unwrap();
        let restored: ParamGroup = serde_json::from_slice(&bytes).unwrap();
        assert!(restored.matches(&ParamId::from(1), Some("layers.2.training")));
        assert!(!restored.matches(&ParamId::from(1), Some("layers.training")));
    }
}
