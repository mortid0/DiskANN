/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Experimental label strategies for graphs constructed with label-specific entry points.
//!
//! Label membership uses external IDs and is immutable. Provider deletions are observed
//! during search. If every construction-selected entry point has disappeared, search
//! starts at the first available member in the original posting order. Deletions can
//! still disconnect a label's graph; this fallback does not repair its connectivity.
//! Concurrent searches and updates do not provide snapshot isolation, and new external
//! IDs do not join an existing label automatically.

use std::{collections::BTreeMap, sync::Arc};

use diskann::{
    ANNError, ANNResult,
    graph::{SearchOutputBuffer, glue, search_output_buffer::BufferState},
    neighbor::Neighbor,
    provider::HasId,
};
use hashbrown::HashMap;

use super::{Context, Id, Provider, SearchAccessor, Translate, ready};
use crate::{ids::IdMap, repr};

/// A construction-time error in label membership or entry points.
#[derive(Debug, thiserror::Error)]
pub enum LabelIndexError {
    #[error("label membership references an external ID absent from the provider")]
    MissingPoint,
    #[error("entry points reference an unknown label")]
    MissingLabel,
    #[error("an entry point is not a member of its label")]
    RejectedEntryPoint,
}

#[derive(Debug)]
struct Label<M> {
    // Preserve posting order for deterministic fallback without a second copy of the IDs.
    members: HashMap<M, usize>,
    entry_points: Vec<M>,
}

impl<M: Id> Label<M> {
    fn accepts(&self, mapping: &IdMap<M>, id: u32) -> bool {
        mapping
            .to_external(id)
            .is_some_and(|external| self.members.contains_key(&external))
    }
}

/// Label membership and construction-selected entry points in external ID space.
#[derive(Debug)]
pub struct LabelIndex<L, M: Id> {
    labels: BTreeMap<L, Arc<Label<M>>>,
    empty: Arc<Label<M>>,
}

impl<L: Ord + Clone, M: Id> LabelIndex<L, M> {
    /// Validate label metadata against a provider after importing its graph.
    ///
    /// # Errors
    /// Returns an error for missing provider IDs, unknown labels in `entry_points`,
    /// or entry points that do not belong to their label. Duplicate IDs are ignored.
    pub fn new<R>(
        provider: &Provider<R, M>,
        postings: &BTreeMap<L, Vec<M>>,
        entry_points: &BTreeMap<L, Vec<M>>,
    ) -> Result<Self, LabelIndexError> {
        if entry_points
            .keys()
            .any(|label| !postings.contains_key(label))
        {
            return Err(LabelIndexError::MissingLabel);
        }
        let mut labels = BTreeMap::new();
        for (label, members) in postings {
            if members
                .iter()
                .any(|id| !provider.mapping.contains_external(id))
            {
                return Err(LabelIndexError::MissingPoint);
            }
            let mut ordered_members = HashMap::new();
            for (position, id) in members.iter().enumerate() {
                ordered_members.entry(id.clone()).or_insert(position);
            }
            let mut selected = Vec::new();
            if let Some(points) = entry_points.get(label) {
                for id in points {
                    if !ordered_members.contains_key(id) {
                        return Err(LabelIndexError::RejectedEntryPoint);
                    }
                    if !selected.contains(id) {
                        selected.push(id.clone());
                    }
                }
            }
            labels.insert(
                label.clone(),
                Arc::new(Label {
                    members: ordered_members,
                    entry_points: selected,
                }),
            );
        }
        Ok(Self {
            labels,
            empty: Arc::new(Label {
                members: HashMap::new(),
                entry_points: Vec::new(),
            }),
        })
    }

    /// Select a label without exposing internal IDs. Unknown labels return no results.
    pub fn strategy(&self, label: &L) -> Strategy<M> {
        Strategy {
            label: self.labels.get(label).unwrap_or(&self.empty).clone(),
        }
    }
}

/// A shareable filtered strategy using construction-selected label entry points.
#[derive(Debug, Clone)]
pub struct Strategy<M: Id> {
    label: Arc<Label<M>>,
}

/// A label-aware adapter around the provider's existing search accessor.
pub struct Accessor<'a, M: Id> {
    inner: SearchAccessor<'a>,
    mapping: &'a IdMap<M>,
    label: &'a Label<M>,
}

impl<M: Id> HasId for Accessor<'_, M> {
    type Id = u32;
}

impl<M: Id> Accessor<'_, M> {
    fn entry_point(&mut self, external: &M) -> ANNResult<Option<Neighbor<u32>>> {
        let Some(id) = self.mapping.to_internal(external) else {
            return Ok(None);
        };
        // A deletion can retire the vector before removing its ID mapping.
        let Some(distance) = self.inner.expand_beam.evaluate(id)? else {
            return Ok(None);
        };
        self.inner.counters.get_vector(1);
        self.inner.counters.query_distance(1);
        Ok((!distance.is_nan() && self.label.accepts(self.mapping, id))
            .then_some(Neighbor::new(id, distance)))
    }
}

impl<'a, R: repr::Search, M: Id> glue::SearchStrategy<'a, Provider<R, M>, R::Query<'a>>
    for Strategy<M>
{
    type SearchAccessor = Accessor<'a, M>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a Provider<R, M>,
        context: &'a Context,
        query: R::Query<'a>,
    ) -> ANNResult<Self::SearchAccessor> {
        Ok(Accessor {
            inner: glue::SearchStrategy::search_accessor(
                &super::Strategy,
                provider,
                context,
                query,
            )?,
            mapping: &provider.mapping,
            label: &self.label,
        })
    }
}

impl<M: Id> glue::FilteredAccessor for Accessor<'_, M> {
    fn start_point_distances<F>(&mut self, mut f: F) -> impl Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(glue::Decision<u32>, f32) + Send,
    {
        ready(move || {
            let mut accepted = false;
            for external in &self.label.entry_points {
                if let Some(point) = self.entry_point(external)? {
                    f(glue::Decision::accept(*point.id()), *point.distance());
                    accepted = true;
                }
            }
            if !accepted {
                let mut fallback: Option<(usize, Neighbor<u32>)> = None;
                for (external, &position) in &self.label.members {
                    if fallback.is_some_and(|(selected, _)| selected < position) {
                        continue;
                    }
                    if let Some(point) = self.entry_point(external)? {
                        fallback = Some((position, point));
                    }
                }
                if let Some((_, point)) = fallback {
                    f(glue::Decision::accept(*point.id()), *point.distance());
                }
            }
            Ok(())
        })
    }

    fn num_starting_points(&self) -> impl Future<Output = ANNResult<usize>> + Send {
        ready(|| {
            let count = self
                .label
                .entry_points
                .iter()
                .filter(|id| self.mapping.contains_external(*id))
                .count();
            Ok(if count != 0 {
                count
            } else {
                usize::from(
                    self.label
                        .members
                        .keys()
                        .any(|id| self.mapping.contains_external(id)),
                )
            })
        })
    }

    fn expand_beam_filtered<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = u32> + Send,
        P: glue::HybridPredicate<u32> + Send + Sync,
        F: FnMut(glue::Decision<u32>, f32) + Send,
    {
        let Self {
            inner,
            mapping,
            label,
        } = self;
        glue::SearchAccessor::expand_beam(inner, ids, pred, move |id, distance| {
            let decision = if label.accepts(mapping, id) {
                glue::Decision::accept(id)
            } else {
                glue::Decision::reject(id)
            };
            on_neighbors(decision, distance);
        })
    }

    fn expand_beam_accept_only<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        mut on_neighbors: F,
    ) -> impl Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = u32> + Send,
        P: glue::Predicate<u32> + glue::PredicateMut<glue::Accept<u32>> + Send + Sync,
        F: FnMut(glue::Accept<u32>, f32) + Send,
    {
        let Self {
            inner,
            mapping,
            label,
        } = self;
        let pred = Accepted {
            pred,
            mapping,
            label,
        };
        glue::SearchAccessor::expand_beam(inner, ids, pred, move |id, distance| {
            if label.accepts(mapping, id) {
                on_neighbors(glue::Accept::new(id), distance);
            }
        })
    }
}

struct Accepted<'a, P, M: Id> {
    pred: P,
    mapping: &'a IdMap<M>,
    label: &'a Label<M>,
}

impl<P: glue::Predicate<u32>, M: Id> glue::Predicate<u32> for Accepted<'_, P, M> {
    fn eval(&self, id: &u32) -> bool {
        self.pred.eval(id) && self.label.accepts(self.mapping, *id)
    }
}

impl<P: glue::PredicateMut<glue::Accept<u32>>, M: Id> glue::PredicateMut<u32>
    for Accepted<'_, P, M>
{
    fn eval_mut(&mut self, id: &u32) -> bool {
        self.label.accepts(self.mapping, *id) && self.pred.eval_mut(&glue::Accept::new(*id))
    }
}

impl<P, M: Id> glue::HybridPredicate<u32> for Accepted<'_, P, M> where
    P: glue::Predicate<u32> + glue::PredicateMut<glue::Accept<u32>>
{
}

/// Delegate reranking and external ID translation to the provider's existing processor.
pub struct PostProcess<R, M>(Translate<R, M>);

impl<'a, R: repr::Search, M: Id> glue::SearchPostProcess<Accessor<'a, M>, R::Query<'a>, M>
    for PostProcess<R, M>
where
    R::Query<'a>: Send,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut Accessor<'a, M>,
        query: R::Query<'a>,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = ANNResult<usize>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<M> + Send + ?Sized,
    {
        let before = output.current_len();
        async move {
            let Accessor { inner, label, .. } = accessor;
            // Filter after ID translation so a concurrently recycled slot cannot return
            // a new external ID outside this label.
            let mut filtered = AcceptedOutput {
                output,
                members: &label.members,
            };
            glue::SearchPostProcess::post_process(&self.0, inner, query, candidates, &mut filtered)
                .await?;
            Ok(filtered.current_len() - before)
        }
    }
}

impl<'a, R: repr::Search, M: Id> glue::DefaultPostProcessor<'a, Provider<R, M>, R::Query<'a>, M>
    for Strategy<M>
where
    R::Query<'a>: Send,
{
    type Processor = PostProcess<R, M>;

    fn default_post_processor(&'a self) -> Self::Processor {
        PostProcess(Translate::default())
    }
}

struct AcceptedOutput<'a, B: ?Sized, M> {
    output: &'a mut B,
    members: &'a HashMap<M, usize>,
}

impl<B: SearchOutputBuffer<M> + ?Sized, M: Id> SearchOutputBuffer<M> for AcceptedOutput<'_, B, M> {
    fn size_hint(&self) -> Option<usize> {
        self.output.size_hint()
    }

    fn current_len(&self) -> usize {
        self.output.current_len()
    }

    fn push(&mut self, neighbor: Neighbor<M>) -> BufferState {
        if self.members.contains_key(neighbor.id()) {
            self.output.push(neighbor)
        } else if self.output.size_hint() == Some(0) {
            BufferState::Full
        } else {
            BufferState::Available
        }
    }

    fn extend<Itr: IntoIterator<Item = Neighbor<M>>>(&mut self, itr: Itr) -> usize {
        self.output.extend(
            itr.into_iter()
                .filter(|n| self.members.contains_key(n.id())),
        )
    }
}

#[cfg(test)]
mod tests;
