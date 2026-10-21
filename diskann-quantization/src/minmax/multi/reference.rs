/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::kernel::{MinMaxMaxSimKernel, validate};
use crate::{
    bits::{Representation, Unsigned},
    minmax::MinMaxMeta,
    multi_vector::{Defaulted, Mat, MatRef, MaxSimError},
};

#[derive(Debug)]
pub(super) struct Reference<const A_BITS: usize>
where
    Unsigned: Representation<A_BITS>,
{
    query: Mat<MinMaxMeta<A_BITS>>,
}

impl<const A_BITS: usize> Reference<A_BITS>
where
    Unsigned: Representation<A_BITS>,
{
    #[expect(
        clippy::expect_used,
        reason = "validated canonical rows contain in-range codes"
    )]
    pub(super) fn new(query: MatRef<'_, MinMaxMeta<A_BITS>>) -> Self {
        let mut owned = match Mat::new(*query.repr(), Defaulted) {
            Ok(value) => value,
            Err(never) => match never {},
        };
        for (source, mut target) in query.rows().zip(owned.rows_mut()) {
            target.set_meta(source.meta());
            for d in 0..query.repr().intrinsic_dim() {
                let value = source.vector().get(d).expect("canonical code in bounds");
                target
                    .vector_mut()
                    .set(d, value)
                    .expect("canonical code in range");
            }
        }
        Self { query: owned }
    }
}

impl<const A_BITS: usize, const B_BITS: usize> MinMaxMaxSimKernel<B_BITS> for Reference<A_BITS>
where
    Unsigned: Representation<A_BITS> + Representation<B_BITS>,
{
    fn nrows(&self) -> usize {
        self.query.num_vectors()
    }

    #[expect(
        clippy::expect_used,
        reason = "dimensions checked before iterating canonical rows"
    )]
    fn compute_max_sim(
        &self,
        document: MatRef<'_, MinMaxMeta<B_BITS>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError> {
        let dim = self.query.repr().intrinsic_dim();
        validate(self.query.num_vectors(), dim, document, scores)?;
        if document.num_vectors() == 0 {
            scores.fill(f32::MAX);
            return Ok(());
        }
        if dim == 0 {
            scores.fill(0.0);
            return Ok(());
        }
        for (query, best) in self.query.rows().zip(scores) {
            *best = f32::MAX;
            let q = query.meta();
            for doc in document.rows() {
                // Independent scalar pairwise algorithm: no prepared panels or SIMD
                // compensation helpers; supports every 1..=8 bit-width combination.
                let dot: u64 = (0..dim)
                    .map(|d| {
                        query.vector().get(d).expect("query code in bounds") as u64
                            * doc.vector().get(d).expect("document code in bounds") as u64
                    })
                    .sum();
                let b = doc.meta();
                let ip =
                    ((q.a * b.a) * dot as f32 + q.n * b.b) + b.n * q.b + (q.b * b.b) * dim as f32;
                *best = best.min(-ip);
            }
        }
        Ok(())
    }
}
