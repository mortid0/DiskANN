/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::Architecture;
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};
use diskann_wide::arch::{Scalar, Target1};

use super::{kernel::MinMaxErase, prepared::Prepared, reference::Reference};
use crate::{
    bits::{Representation, Unsigned},
    matrix_kernels::maxsim::packed_minmax_x_unpacked_minmax::max_k,
    minmax::{MinMaxCompensation, MinMaxMeta},
    multi_vector::{MatRef, MaxSimIsa, NotSupported, Overflow},
};

/// Failure to prepare a MinMax MaxSim query.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[non_exhaustive]
pub enum BuildMinMaxMaxSimError {
    #[error(transparent)]
    NotSupported(#[from] NotSupported),
    #[error("query dimension {0} exceeds the integer accumulator limit of {1}")]
    DimTooLarge(usize, usize),
    #[error(transparent)]
    Overflow(#[from] Overflow),
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::bits::Unsigned {}
}

/// Builds a prepared MinMax kernel for a fixed document bit width.
///
/// Implemented for [`Unsigned`] at widths 1 through 8. The sealed implementations
/// select a supported ISA and numeric layout without instantiating invalid combinations.
/// Generic callers use this bound when forwarding to [`build_minmax_max_sim`].
pub trait MinMaxKernelFactory<const B_BITS: usize>:
    Representation<B_BITS> + sealed::Sealed
{
    #[doc(hidden)]
    fn build<const A_BITS: usize, E: MinMaxErase<B_BITS>>(
        isa: MaxSimIsa,
        query: MatRef<'_, MinMaxMeta<A_BITS>>,
        erase: E,
    ) -> Result<E::Output, BuildMinMaxMaxSimError>
    where
        Unsigned: Representation<A_BITS> + Representation<B_BITS>;
}

struct BuildAndErase<const A_BITS: usize, const B_BITS: usize, E>(E);

// Implement construction and type erasure for one concrete ISA and panel layout.
macro_rules! kernel_builder {
    ($b:literal, $arch:ty, $mr:literal, $nr:literal, $pack:literal) => {
        impl<const A_BITS: usize, E: MinMaxErase<$b>>
            Target1<$arch, Result<E::Output, Overflow>, MatRef<'_, MinMaxMeta<A_BITS>>>
            for BuildAndErase<A_BITS, $b, E>
        where
            Unsigned: Representation<A_BITS>,
        {
            fn run(
                self,
                arch: $arch,
                query: MatRef<'_, MinMaxMeta<A_BITS>>,
            ) -> Result<E::Output, Overflow> {
                let kernel = Prepared::<$arch, A_BITS, $mr, $nr, $pack>::new(arch, query)?;
                Ok(self.0.erase(kernel))
            }
        }
    };
}

// Implement the factory for one document bit width and register its ISA layouts.
macro_rules! kernel_factory {
    (
        b_bits: $b:literal,
        scalar: { mr: $scalar_mr:literal, pack: $scalar_pack:literal },
        v3_pack: $v3_pack:literal,
        v4_pack: $v4_pack:literal $(,)?
    ) => {
        kernel_builder!($b, Scalar, $scalar_mr, 2, $scalar_pack);
        #[cfg(target_arch = "x86_64")]
        kernel_builder!($b, V3, 16, 4, $v3_pack);
        #[cfg(target_arch = "x86_64")]
        kernel_builder!($b, V4, 16, 8, $v4_pack);
        #[cfg(target_arch = "aarch64")]
        kernel_builder!($b, Neon, 8, 4, 4);

        impl MinMaxKernelFactory<$b> for Unsigned {
            fn build<const A_BITS: usize, E: MinMaxErase<$b>>(
                isa: MaxSimIsa,
                query: MatRef<'_, MinMaxMeta<A_BITS>>,
                erase: E,
            ) -> Result<E::Output, BuildMinMaxMaxSimError>
            where
                Unsigned: Representation<A_BITS>,
            {
                validate_query::<A_BITS, $b>(query)?;
                let builder = BuildAndErase::<A_BITS, $b, E>(erase);
                let unavailable = || NotSupported {
                    isa,
                    reason: "requested ISA unavailable on this host",
                };
                let built = match isa {
                    MaxSimIsa::Auto => diskann_wide::arch::dispatch1_no_features(builder, query),
                    MaxSimIsa::Scalar => Scalar::new().run1(builder, query),
                    #[cfg(target_arch = "x86_64")]
                    MaxSimIsa::X86_64_V3 => V3::new_checked()
                        .ok_or_else(unavailable)?
                        .run1(builder, query),
                    #[cfg(target_arch = "x86_64")]
                    MaxSimIsa::X86_64_V4 => V4::new_checked()
                        .ok_or_else(unavailable)?
                        .run1(builder, query),
                    #[cfg(not(target_arch = "x86_64"))]
                    MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => return Err(unavailable().into()),
                    #[cfg(target_arch = "aarch64")]
                    MaxSimIsa::Neon => Neon::new_checked()
                        .ok_or_else(unavailable)?
                        .run1(builder, query),
                    #[cfg(not(target_arch = "aarch64"))]
                    MaxSimIsa::Neon => return Err(unavailable().into()),
                    MaxSimIsa::Reference => return Ok(builder.0.erase(Reference::new(query))),
                };
                Ok(built?)
            }
        }
    };
}

// Literal document widths prevent unsupported byte paths from being monomorphized.
// V3's maddubs intermediate limits B to six bits; signed bytes limit Scalar/V4 to seven.
kernel_factory!(
    b_bits: 1,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 2,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 3,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 4,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 5,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 6,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 7,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 2,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 8,
    scalar: { mr: 8, pack: 2 },
    v3_pack: 2,
    v4_pack: 2,
);

/// Prepare an owned MinMax query, automatically selecting a legal numeric layout.
///
/// `Auto` chooses an available ISA once. Explicit unavailable ISAs return an error;
/// `Reference` uses a scalar pairwise algorithm. V3 uses PACK4 for B1..6 and PACK2
/// for B7..8; V4 uses PACK4 for B1..7 and PACK2 for B8. Layout parameters stay internal.
///
/// Scores follow the existing MaxSim distance convention (negative maximum inner
/// product), not the internal matrix kernel's unnegated inner products.
///
/// # Errors
///
/// Returns [`BuildMinMaxMaxSimError`] for unsupported ISA, a dimension beyond the
/// common i32 accumulator limit, or an unrepresentable canonical/prepared extent.
/// These checks also apply to `Reference` and empty queries. Allocation exhaustion
/// follows the standard allocator's behavior.
///
/// # Example
///
/// ```
/// use diskann_quantization::{
///     minmax::{MinMaxMeta, build_minmax_max_sim},
///     multi_vector::{BoxErase, Defaulted, Mat, MaxSimIsa},
/// };
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let query = Mat::new(MinMaxMeta::<8>::new(2, 16), Defaulted)?;
/// let document = Mat::new(MinMaxMeta::<4>::new(3, 16), Defaulted)?;
/// let kernel = build_minmax_max_sim::<8, 4, _>(MaxSimIsa::Auto, query.as_view(), BoxErase)?;
/// let mut scores = vec![0.0; kernel.nrows()];
/// kernel.compute_max_sim(document.as_view(), &mut scores)?;
/// assert_eq!(scores, vec![0.0; 2]);
/// # Ok(())
/// # }
/// # example().unwrap();
/// ```
pub fn build_minmax_max_sim<const A_BITS: usize, const B_BITS: usize, E>(
    isa: MaxSimIsa,
    query: MatRef<'_, MinMaxMeta<A_BITS>>,
    erase: E,
) -> Result<E::Output, BuildMinMaxMaxSimError>
where
    Unsigned: Representation<A_BITS> + MinMaxKernelFactory<B_BITS>,
    E: MinMaxErase<B_BITS>,
{
    <Unsigned as MinMaxKernelFactory<B_BITS>>::build(isa, query, erase)
}

fn validate_query<const A_BITS: usize, const B_BITS: usize>(
    query: MatRef<'_, MinMaxMeta<A_BITS>>,
) -> Result<(), BuildMinMaxMaxSimError>
where
    Unsigned: Representation<A_BITS> + Representation<B_BITS>,
{
    let dim = query.repr().intrinsic_dim();
    let limit = max_k::<A_BITS, B_BITS>();
    if dim > limit {
        return Err(BuildMinMaxMaxSimError::DimTooLarge(dim, limit));
    }
    let rows = query.num_vectors();
    let overflow = || Overflow::for_type::<u8>(rows, dim);
    let code_bytes = dim.checked_mul(A_BITS).ok_or_else(overflow)?.div_ceil(8);
    let row_bytes = code_bytes
        .checked_add(size_of::<MinMaxCompensation>())
        .ok_or_else(overflow)?;
    Overflow::check::<u8>(rows, row_bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        minmax::{MinMaxCompensation, MinMaxMaxSimKernel},
        multi_vector::{BoxErase, Defaulted, Mat, MaxSimError},
    };

    fn matrix<const BITS: usize>(rows: usize, dim: usize) -> Mat<MinMaxMeta<BITS>>
    where
        Unsigned: Representation<BITS>,
    {
        let mut mat = Mat::new(MinMaxMeta::new(rows, dim), Defaulted).unwrap();
        for (i, mut row) in mat.rows_mut().enumerate() {
            let mut sum = 0;
            for d in 0..dim {
                let code = ((i * 17 + d * 29 + 7) % (1 << BITS)) as i64;
                row.vector_mut().set(d, code).unwrap();
                sum += code;
            }
            let a = (i + 1) as f32 / 8.0;
            row.set_meta(MinMaxCompensation {
                dim: dim as u32,
                a,
                b: i as f32 / 4.0 - 1.0,
                n: a * sum as f32,
                norm_squared: 0.0,
            });
        }
        mat
    }

    fn isas() -> impl Iterator<Item = MaxSimIsa> {
        [
            MaxSimIsa::Scalar,
            MaxSimIsa::Auto,
            MaxSimIsa::X86_64_V3,
            MaxSimIsa::X86_64_V4,
            MaxSimIsa::Neon,
            MaxSimIsa::Reference,
        ]
        .into_iter()
        .filter(|isa| isa.is_available())
    }

    fn check<const A: usize, const B: usize>(rows: usize, docs: usize, dim: usize)
    where
        Unsigned: Representation<A> + MinMaxKernelFactory<B>,
    {
        let query = matrix::<A>(rows, dim);
        let document = matrix::<B>(docs, dim);
        let reference =
            build_minmax_max_sim::<A, B, _>(MaxSimIsa::Reference, query.as_view(), BoxErase)
                .unwrap();
        let mut expected = vec![0.0; rows];
        reference
            .compute_max_sim(document.as_view(), &mut expected)
            .unwrap();
        for isa in isas() {
            let kernel = build_minmax_max_sim::<A, B, _>(isa, query.as_view(), BoxErase).unwrap();
            assert_eq!(kernel.nrows(), rows);
            let mut guarded = vec![123.0; rows + 2];
            kernel
                .compute_max_sim(document.as_view(), &mut guarded[1..rows + 1])
                .unwrap();
            for (actual, expected) in guarded[1..rows + 1].iter().zip(&expected) {
                assert!(
                    actual == expected
                        || (actual.is_finite()
                            && expected.is_finite()
                            && (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0)),
                    "{A}x{B}, {isa}: {actual} != {expected}"
                );
            }
            assert_eq!((guarded[0], guarded[rows + 1]), (123.0, 123.0));
        }
    }

    #[test]
    fn all_bit_width_combinations_match_pairwise() {
        macro_rules! row {
            ($a:literal) => {
                check::<$a, 1>(3, 5, 5);
                check::<$a, 2>(3, 5, 5);
                check::<$a, 3>(3, 5, 5);
                check::<$a, 4>(3, 5, 5);
                check::<$a, 5>(3, 5, 5);
                check::<$a, 6>(3, 5, 5);
                check::<$a, 7>(3, 5, 5);
                check::<$a, 8>(3, 5, 5);
            };
        }
        row!(1);
        row!(2);
        row!(3);
        row!(4);
        row!(5);
        row!(6);
        row!(7);
        row!(8);
    }

    #[test]
    fn tails_and_empty_inputs() {
        for (q, d, k) in [(17, 9, 65), (0, 5, 7), (3, 0, 7), (3, 5, 0), (3, 0, 0)] {
            check::<8, 4>(q, d, k);
            check::<8, 8>(q, d, k);
        }
    }

    #[test]
    fn errors_leave_output_untouched_and_isa_is_not_silently_downgraded() {
        let query = matrix::<8>(2, 5);
        for isa in isas() {
            let kernel = build_minmax_max_sim::<8, 8, _>(isa, query.as_view(), BoxErase).unwrap();
            let doc = matrix::<8>(1, 6);
            let mut scores = [123.0; 2];
            assert!(matches!(
                kernel.compute_max_sim(doc.as_view(), &mut scores),
                Err(MaxSimError::UnequalDim(6, 5))
            ));
            assert_eq!(scores, [123.0; 2]);
            let doc = matrix::<8>(1, 5);
            let mut scores = [123.0];
            assert!(matches!(
                kernel.compute_max_sim(doc.as_view(), &mut scores),
                Err(MaxSimError::InvalidBufferLength(1, 2))
            ));
            assert_eq!(scores, [123.0]);
        }
        for isa in [MaxSimIsa::X86_64_V3, MaxSimIsa::X86_64_V4, MaxSimIsa::Neon] {
            if !isa.is_available() {
                assert!(matches!(
                    build_minmax_max_sim::<8, 8, _>(isa, query.as_view(), BoxErase),
                    Err(BuildMinMaxMaxSimError::NotSupported(_))
                ));
            }
        }
        let dim = max_k::<8, 8>() + 1;
        let query = MatRef::new(MinMaxMeta::<8>::new(0, dim), &[] as &[u8]).unwrap();
        for isa in [MaxSimIsa::Auto, MaxSimIsa::Reference] {
            assert!(
                matches!(build_minmax_max_sim::<8,8,_>(isa,query,BoxErase),Err(BuildMinMaxMaxSimError::DimTooLarge(d,_)) if d==dim)
            );
        }
    }

    #[test]
    fn prepared_query_owns_data_and_scores_use_distance_semantics() {
        for isa in isas() {
            let kernel = {
                let mut query = matrix::<8>(2, 3);
                for mut row in query.rows_mut() {
                    for d in 0..3 {
                        row.vector_mut().set(d, 1).unwrap();
                    }
                    row.set_meta(MinMaxCompensation {
                        dim: 3,
                        a: 1.0,
                        b: 0.0,
                        n: 3.0,
                        norm_squared: 0.0,
                    });
                }
                build_minmax_max_sim::<8, 4, _>(isa, query.as_view(), BoxErase).unwrap()
            };
            let mut doc = matrix::<4>(2, 3);
            for mut row in doc.rows_mut() {
                for d in 0..3 {
                    row.vector_mut().set(d, 2).unwrap();
                }
                row.set_meta(MinMaxCompensation {
                    dim: 3,
                    a: 1.0,
                    b: 0.0,
                    n: 6.0,
                    norm_squared: 0.0,
                });
            }
            let mut scores = [0.0; 2];
            kernel.compute_max_sim(doc.as_view(), &mut scores).unwrap();
            assert_eq!(scores, [-6.0; 2]);
            for mut row in doc.rows_mut() {
                row.set_meta(MinMaxCompensation {
                    dim: 3,
                    a: f32::NAN,
                    ..Default::default()
                });
            }
            kernel.compute_max_sim(doc.as_view(), &mut scores).unwrap();
            assert_eq!(scores, [f32::MAX; 2]);
            let mut row = doc.get_row_mut(0).unwrap();
            row.set_meta(MinMaxCompensation {
                dim: 3,
                a: 1.0,
                b: 0.0,
                n: 6.0,
                norm_squared: 0.0,
            });
            kernel.compute_max_sim(doc.as_view(), &mut scores).unwrap();
            assert_eq!(scores, [-6.0; 2]);
        }
    }

    struct Identify;
    impl<const B: usize> MinMaxErase<B> for Identify
    where
        Unsigned: Representation<B>,
    {
        type Output = std::any::TypeId;
        fn erase<K: MinMaxMaxSimKernel<B> + 'static>(self, _: K) -> Self::Output {
            std::any::TypeId::of::<K>()
        }
    }

    #[test]
    fn factory_selects_numeric_layouts_before_packing() {
        let query = matrix::<8>(1, 3);
        assert_eq!(
            build_minmax_max_sim::<8, 7, _>(MaxSimIsa::Scalar, query.as_view(), Identify).unwrap(),
            std::any::TypeId::of::<Prepared<Scalar, 8, 16, 2, 4>>()
        );
        assert_eq!(
            build_minmax_max_sim::<8, 8, _>(MaxSimIsa::Scalar, query.as_view(), Identify).unwrap(),
            std::any::TypeId::of::<Prepared<Scalar, 8, 8, 2, 2>>()
        );
        #[cfg(target_arch = "x86_64")]
        {
            if V3::new_checked().is_some() {
                assert_eq!(
                    build_minmax_max_sim::<8, 6, _>(
                        MaxSimIsa::X86_64_V3,
                        query.as_view(),
                        Identify
                    )
                    .unwrap(),
                    std::any::TypeId::of::<Prepared<V3, 8, 16, 4, 4>>()
                );
                assert_eq!(
                    build_minmax_max_sim::<8, 7, _>(
                        MaxSimIsa::X86_64_V3,
                        query.as_view(),
                        Identify
                    )
                    .unwrap(),
                    std::any::TypeId::of::<Prepared<V3, 8, 16, 4, 2>>()
                );
                assert_eq!(
                    build_minmax_max_sim::<8, 8, _>(
                        MaxSimIsa::X86_64_V3,
                        query.as_view(),
                        Identify
                    )
                    .unwrap(),
                    std::any::TypeId::of::<Prepared<V3, 8, 16, 4, 2>>()
                );
            }
            if V4::new_checked().is_some() {
                assert_eq!(
                    build_minmax_max_sim::<8, 7, _>(
                        MaxSimIsa::X86_64_V4,
                        query.as_view(),
                        Identify
                    )
                    .unwrap(),
                    std::any::TypeId::of::<Prepared<V4, 8, 16, 8, 4>>()
                );
                assert_eq!(
                    build_minmax_max_sim::<8, 8, _>(
                        MaxSimIsa::X86_64_V4,
                        query.as_view(),
                        Identify
                    )
                    .unwrap(),
                    std::any::TypeId::of::<Prepared<V4, 8, 16, 8, 2>>()
                );
            }
        }
    }

    #[test]
    fn factory_distances_match_existing_minmax_pairwise_api() {
        use crate::multi_vector::{MaxSim, QueryMatRef};
        use diskann_vector::DistanceFunctionMut;
        for (rows, docs, dim) in [(3, 5, 5), (17, 9, 65), (3, 0, 5)] {
            let query = matrix::<8>(rows, dim);
            let document = matrix::<4>(docs, dim);
            let mut expected = vec![0.0; rows];
            MaxSim::new(&mut expected)
                .evaluate(QueryMatRef::from(query.as_view()), document.as_view());
            for isa in isas() {
                let kernel =
                    build_minmax_max_sim::<8, 4, _>(isa, query.as_view(), BoxErase).unwrap();
                let mut actual = vec![0.0; rows];
                kernel
                    .compute_max_sim(document.as_view(), &mut actual)
                    .unwrap();
                for (a, b) in actual.iter().zip(&expected) {
                    assert!(
                        a == b
                            || (a.is_finite()
                                && b.is_finite()
                                && (a - b).abs() <= 1e-5 * b.abs().max(1.0))
                    );
                }
            }
        }
    }

    #[test]
    fn document_width_dispatch_cannot_bypass_input_validation() {
        let dim = max_k::<8, 8>() + 1;
        let query = MatRef::new(MinMaxMeta::<8>::new(0, dim), &[] as &[u8]).unwrap();
        assert!(matches!(
            <Unsigned as MinMaxKernelFactory<8>>::build(MaxSimIsa::Reference, query, BoxErase),
            Err(BuildMinMaxMaxSimError::DimTooLarge(_, _))
        ));
    }

    #[test]
    fn factory_handles_byte_boundaries_and_nonfinite_distances() {
        fn check<const B: usize>()
        where
            Unsigned: MinMaxKernelFactory<B>,
        {
            let mut query = matrix::<8>(1, 5);
            let mut document = matrix::<B>(1, 5);
            {
                let mut q = query.get_row_mut(0).unwrap();
                let mut d = document.get_row_mut(0).unwrap();
                for i in 0..5 {
                    q.vector_mut().set(i, 255).unwrap();
                    d.vector_mut().set(i, (1 << B) - 1).unwrap();
                }
                q.set_meta(MinMaxCompensation {
                    dim: 5,
                    a: 1.0,
                    b: 0.0,
                    n: 1275.0,
                    norm_squared: 0.0,
                });
                d.set_meta(MinMaxCompensation {
                    dim: 5,
                    a: 1.0,
                    b: 0.0,
                    n: 5.0 * ((1 << B) - 1) as f32,
                    norm_squared: 0.0,
                });
            }
            for isa in isas() {
                let kernel =
                    build_minmax_max_sim::<8, B, _>(isa, query.as_view(), BoxErase).unwrap();
                let mut scores = [0.0];
                for (scale, expected) in [
                    (1.0, -(5 * 255 * ((1 << B) - 1)) as f32),
                    (f32::NEG_INFINITY, f32::MAX),
                    (f32::INFINITY, f32::NEG_INFINITY),
                    (f32::NAN, f32::MAX),
                ] {
                    let mut row = document.get_row_mut(0).unwrap();
                    let mut meta = row.meta();
                    meta.a = scale;
                    row.set_meta(meta);
                    kernel
                        .compute_max_sim(document.as_view(), &mut scores)
                        .unwrap();
                    assert_eq!(scores, [expected], "{isa}, B={B}");
                }
            }
        }
        check::<6>();
        check::<7>();
        check::<8>();
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn overflowing_canonical_query_is_rejected_before_row_access() {
        // MinMaxMeta itself permits unchecked extents. For K=0 each canonical row
        // occupies 20 bytes; this chosen row count wraps the legacy total to zero.
        let rows = 1usize << (usize::BITS - 2);
        let query = MatRef::new(MinMaxMeta::<8>::new(rows, 0), &[] as &[u8]).unwrap();
        for isa in [MaxSimIsa::Auto, MaxSimIsa::Reference] {
            assert!(matches!(
                build_minmax_max_sim::<8, 4, _>(isa, query, BoxErase),
                Err(BuildMinMaxMaxSimError::Overflow(_))
            ));
        }
    }
}
