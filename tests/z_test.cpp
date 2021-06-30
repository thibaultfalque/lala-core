// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "thrust/optional.h"
#include "ast.hpp"
#include "z.hpp"
#include "allocator.hpp"
#include "utility.hpp"

using namespace lala;

typedef ZInc<int, StandardAllocator> zi;
typedef ZDec<int, StandardAllocator> zd;
typedef Formula<StandardAllocator> F;

TEST(ZDeathTest, BadConstruction) {
  ASSERT_DEATH(zi(Limits<int>::bot()), "");
  ASSERT_DEATH(zi(Limits<int>::top()), "");
  // Dual
  ASSERT_DEATH(zd(Limits<int>::bot()), "");
  ASSERT_DEATH(zd(Limits<int>::top()), "");
}

template<typename VarDom>
void test_formula(Approx appx, const F& f, thrust::optional<VarDom> expect) {
  thrust::optional<VarDom> j = VarDom::bot().interpret(appx, f);
  EXPECT_EQ(j.has_value(), expect.has_value());
  EXPECT_EQ(j, expect);
}

template<typename VarDom>
void test_interpret(F::Type relation, Approx appx, typename VarDom::ValueType elem, thrust::optional<VarDom> expect) {
  test_formula<VarDom>(
    appx,
    make_x_op_i(relation, 0, elem, standard_allocator),
    expect);
}

template<typename VarDom>
void test_all_interpret(F::Type relation, typename VarDom::ValueType elem, thrust::optional<VarDom> expect) {
  Approx appxs[3] = {EXACT, UNDER, OVER};
  for(int i = 0; i < 3; ++i) {
    test_interpret<VarDom>(relation, appxs[i], elem, expect);
  }
}

template<typename VarDom>
void test_exact_interpret(F::Type relation, typename VarDom::ValueType elem, thrust::optional<VarDom> expect) {
  test_interpret<VarDom>(relation, EXACT, elem, expect);
}

template<typename VarDom>
void test_under_interpret(F::Type relation, typename VarDom::ValueType elem, thrust::optional<VarDom> expect) {
  test_interpret<VarDom>(relation, UNDER, elem, expect);
}

template<typename VarDom>
void test_over_interpret(F::Type relation, typename VarDom::ValueType elem, thrust::optional<VarDom> expect) {
  test_interpret<VarDom>(relation, OVER, elem, expect);
}

TEST(ZTest, ValidInterpret) {
  test_all_interpret<zi>(F::GEQ, 10, zi(10));
  test_all_interpret<zi>(F::GT, 10, zi(11));
  test_under_interpret<zi>(F::NEQ, 10, zi(11));
  test_over_interpret<zi>(F::EQ, 10, zi(10));
  // Dual
  test_all_interpret<zd>(F::LEQ, 10, zd(10));
  test_all_interpret<zd>(F::LT, 10, zd(9));
  test_under_interpret<zd>(F::NEQ, 10, zd(9));
  test_over_interpret<zd>(F::EQ, 10, zd(10));
}

TEST(ZTest, NoInterpret) {
  test_exact_interpret<zi>(F::NEQ, 10, {});
  test_exact_interpret<zi>(F::EQ, 10, {});
  test_all_interpret<zi>(F::LEQ, 10, {});
  test_all_interpret<zi>(F::LT, 10, {});
  // Dual
  test_exact_interpret<zd>(F::NEQ, 10, {});
  test_exact_interpret<zd>(F::EQ, 10, {});
  test_all_interpret<zd>(F::GEQ, 10, {});
  test_all_interpret<zd>(F::GT, 10, {});
}

// `a` and `b` are supposed ordered and `a <= b`.
template <typename A>
void join_meet_generic_test(A a, A b) {
  // Reflexivity
  EXPECT_EQ(a.join(a), a);
  EXPECT_EQ(a.meet(a), a);
  EXPECT_EQ(b.join(b), b);
  EXPECT_EQ(b.meet(b), b);
  // Coherency of join/meet w.r.t. ordering
  EXPECT_EQ(a.join(b), b);
  EXPECT_EQ(b.join(a), b);
  // Commutativity
  EXPECT_EQ(a.meet(b), a);
  EXPECT_EQ(b.meet(a), a);
  // Absorbing
  EXPECT_EQ(a.meet(A::top()), a);
  EXPECT_EQ(b.meet(A::top()), b);
  EXPECT_EQ(a.join(A::top()), A::top());
  EXPECT_EQ(b.join(A::top()), A::top());
  EXPECT_EQ(a.meet(A::bot()), A::bot());
  EXPECT_EQ(b.meet(A::bot()), A::bot());
  EXPECT_EQ(a.join(A::bot()), a);
  EXPECT_EQ(b.join(A::bot()), b);
}

TEST(ZTest, JoinMeet) {
  join_meet_generic_test(zi::bot(), zi::top());
  join_meet_generic_test(zi(0), zi(1));
  join_meet_generic_test(zi(-10), zi(10));
  join_meet_generic_test(zi(Limits<int>::top() - 1), zi::top());
  // Dual
  join_meet_generic_test(zd::bot(), zd::top());
  join_meet_generic_test(zd(1), zd(0));
  join_meet_generic_test(zd(10), zd(-10));
  join_meet_generic_test(zd(Limits<int>::bot() + 1), zd::top());
}

TEST(ZTest, Refine) {
  EXPECT_EQ(zi(0).refine(), false);
  EXPECT_EQ(zi::top().refine(), false);
  EXPECT_EQ(zi::bot().refine(), false);
  // Dual
  EXPECT_EQ(zd(0).refine(), false);
  EXPECT_EQ(zd::top().refine(), false);
  EXPECT_EQ(zd::bot().refine(), false);
}

template<typename A>
void generic_entailment_test(A element) {
  EXPECT_EQ(element.entailment(A::top()), false);
  EXPECT_EQ(element.entailment(A::bot()), true);
  EXPECT_EQ(A::bot().entailment(A::bot()), true);
  EXPECT_EQ(A::top().entailment(A::top()), true);
  EXPECT_EQ(A::top().entailment(A::bot()), true);
  EXPECT_EQ(A::top().entailment(element), true);
}

TEST(ZTest, Entailment) {
  EXPECT_EQ(zi(0).entailment(zi(0)), true);
  EXPECT_EQ(zi(1).entailment(zi(0)), true);
  EXPECT_EQ(zi(0).entailment(zi(1)), false);
  EXPECT_EQ(zi(0).entailment(zi(-1)), true);
  EXPECT_EQ(zi(-1).entailment(zi(0)), false);
  generic_entailment_test(zi(0));
  // Dual
  EXPECT_EQ(zd(0).entailment(zd(0)), true);
  EXPECT_EQ(zd(1).entailment(zd(0)), false);
  EXPECT_EQ(zd(0).entailment(zd(1)), true);
  EXPECT_EQ(zd(0).entailment(zd(-1)), false);
  EXPECT_EQ(zd(-1).entailment(zd(0)), true);
  generic_entailment_test(zd(0));
}

template<typename A>
using SplitSeq = DArray<A, StandardAllocator>;

template<typename A>
SplitSeq<A> make_singleton(A x) {
  return SplitSeq<A>({x});
}

template<typename A>
SplitSeq<A> make_empty() {
  return SplitSeq<A>();
}

template<typename A>
void generic_split_test(A element) {
  EXPECT_EQ(element.split(), make_singleton(element));
  EXPECT_EQ(A::top().split(), make_empty<A>());
  EXPECT_EQ(A::bot().split(), make_singleton(A::bot()));
}

TEST(ZTest, Split) {
  generic_split_test(zi(0));
  generic_split_test(zd(0));
}

template<typename A>
void generic_deinterpret_test() {
  EXPECT_EQ(A::bot().deinterpret(), F::make_true());
  EXPECT_EQ(A::top().deinterpret(), F::make_false());
}

TEST(ZTest, Deinterpret) {
  F f10 = make_x_op_i(F::GEQ, 0, 10, standard_allocator);
  zi z10 = zi::bot().interpret(EXACT, f10).value();
  F f10_bis = z10.deinterpret();
  EXPECT_EQ(f10, f10_bis);
  F f9 = make_x_op_i(F::GT, 0, 9, standard_allocator);
  zi z9 = zi::bot().interpret(EXACT, f9).value();
  F f9_bis = z9.deinterpret();
  EXPECT_EQ(f10, f9_bis);
  generic_deinterpret_test<zi>();
  // Dual
  F f10_d = make_x_op_i(F::LEQ, 0, 10, standard_allocator);
  zd z10_d = zd::bot().interpret(EXACT, f10_d).value();
  F f10_bis_d = z10_d.deinterpret();
  EXPECT_EQ(f10_d, f10_bis_d);
  F f11_d = make_x_op_i(F::LT, 0, 11, standard_allocator);
  zd z11_d = zd::bot().interpret(EXACT, f11_d).value();
  F f11_bis_d = z11_d.deinterpret();
  EXPECT_EQ(f10_d, f11_bis_d);
  generic_deinterpret_test<zd>();
}
