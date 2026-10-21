/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use super::kernel::{MinMaxMaxSimKernel, validate};
use crate::{
    bits::{Representation, Unsigned},
    matrix_kernels::{
        self as mk,
        maxsim::packed_minmax_x_unpacked_minmax::{Driver, ExtraWide, PackedA},
    },
    minmax::MinMaxMeta,
    multi_vector::{MatRef, MaxSimError, Overflow},
};

#[derive(Debug)]
pub(super) struct Prepared<
    W: ExtraWide<MR, PACK>,
    const A_BITS: usize,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> where
    Unsigned: Representation<A_BITS>,
{
    arch: W,
    query: PackedA<W::AElem, A_BITS, MR, PACK>,
    rows: usize,
    dim: usize,
}

impl<
    W: ExtraWide<MR, PACK>,
    const A_BITS: usize,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> Prepared<W, A_BITS, MR, NR, PACK>
where
    Unsigned: Representation<A_BITS>,
{
    pub(super) fn new(arch: W, query: MatRef<'_, MinMaxMeta<A_BITS>>) -> Result<Self, Overflow> {
        Ok(Self {
            arch,
            query: PackedA::new(query)?,
            rows: query.num_vectors(),
            dim: query.repr().intrinsic_dim(),
        })
    }
}

impl<
    W,
    const A_BITS: usize,
    const B_BITS: usize,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> MinMaxMaxSimKernel<B_BITS> for Prepared<W, A_BITS, MR, NR, PACK>
where
    W: ExtraWide<MR, PACK> + Send + Sync + std::fmt::Debug,
    W::AElem: Send + Sync + std::fmt::Debug,
    Unsigned: Representation<A_BITS> + Representation<B_BITS>,
    for<'a> Driver<'a, W, B_BITS, MR, NR, PACK>: mk::Drive,
{
    fn nrows(&self) -> usize {
        self.rows
    }

    fn compute_max_sim(
        &self,
        document: MatRef<'_, MinMaxMeta<B_BITS>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError> {
        validate(self.rows, self.dim, document, scores)?;
        if self.rows == 0 {
            return Ok(());
        }
        if document.num_vectors() == 0 {
            scores.fill(f32::MAX);
            return Ok(());
        }
        let Some(k) = NonZeroUsize::new(self.dim).map(mk::DimK::new) else {
            scores.fill(0.0);
            return Ok(());
        };
        // SAFETY: Construction validates the integer extent. Checks above establish
        // nonempty A, positive K, equal dimensions and exact output coverage.
        let mut driver = unsafe {
            Driver::<_, B_BITS, MR, NR, PACK>::new(
                self.arch,
                &self.query,
                document,
                scores,
                k,
                mk::Cache::detect(),
            )
        };
        mk::Drive::drive(&mut driver);
        // Match the existing MinMax pairwise distance's f32::MAX initial value,
        // including no match and a valid negative-infinite inner product.
        for score in scores {
            *score = (-*score).min(f32::MAX);
        }
        Ok(())
    }
}
