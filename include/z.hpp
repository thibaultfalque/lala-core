// Copyright 2021 Pierre Talbot

#ifndef Z_HPP
#define Z_HPP

#include <type_traits>
#include <utility>
#include <cmath>
#include "thrust/optional.h"
#include "utility.hpp"
#include "darray.hpp"
#include "ast.hpp"

namespace lala {

enum Sign {
  SNEG,
  SPOS,
  SIGNED,
  BOUNDED
};

template<class VT, Sign sign>
struct ZIncUniverse;

template<class VT, Sign sign = SIGNED>
struct ZDecUniverse {
  constexpr static bool increasing = false;
  constexpr static bool decreasing = true;
  using dual_type = ZIncUniverse<VT, sign>;
  using ValueType = VT;
  static ValueType next(ValueType i) {
    if(i == top() || (i == bot() && (sign == SIGNED || sign == SPOS))) {
      return i;
    }
    else {
      return i - 1;
    }
  }
  static ValueType bot() {
    if constexpr (sign == SNEG) {
      return ValueType{};
    }
    else {
      return Limits<ValueType>::top();
    }
  }
  static ValueType top() {
    if constexpr (sign == SPOS) {
      return ValueType{};
    }
    else {
      return Limits<ValueType>::bot();
    }
  }
  static ValueType join(ValueType x, ValueType y) { return min(x, y); }
  static ValueType meet(ValueType x, ValueType y) { return max(x, y); }
  static bool order(ValueType x, ValueType y) { return x >= y; }
  static bool strict_order(ValueType x, ValueType y) { return x > y; }
  static Sig sig_order() { return LEQ; }
  static Sig sig_strict_order() { return LT; }
  static void check(ValueType i) {
    if constexpr(sign == SIGNED) {
      assert(strict_order(bot(), i) && strict_order(i, top()));
    }
    else if constexpr(sign == SNEG) {
      assert(order(top(), i) && strict_order(i, bot()));
    }
    else if constexpr(sign == SPOS) {
      assert(strict_order(bot(), i) && order(i, top()));
    }
  }
};

template<class VT, Sign sign = SIGNED>
struct ZIncUniverse {
  constexpr static bool increasing = true;
  constexpr static bool decreasing = false;
  using dual_type = ZDecUniverse<VT, sign>;
  using ValueType = VT;
  static ValueType next(ValueType i) {
    if(i == top() || (i == bot() && (sign == SIGNED || sign == SNEG))) {
      return i;
    }
    return i + 1;
  }
  static ValueType bot() {
    if constexpr (sign == SPOS) {
      return ValueType{};
    }
    else {
      return Limits<ValueType>::bot();
    }
  }
  static ValueType top() {
    if constexpr (sign == SNEG) {
      return ValueType{};
    }
    else {
      return Limits<ValueType>::top();
    }
  }
  static ValueType join(ValueType x, ValueType y) { return max(x, y); }
  static ValueType meet(ValueType x, ValueType y) { return min(x, y); }
  static bool order(ValueType x, ValueType y) { return x <= y; }
  static bool strict_order(ValueType x, ValueType y) { return x < y; }
  static Sig sig_order() { return GEQ; }
  static Sig sig_strict_order() { return GT; }
  static void check(ValueType i) {
    if constexpr(sign == SIGNED) {
      assert(strict_order(bot(), i) && strict_order(i, top()));
    }
    else if constexpr(sign == SNEG) {
      assert(strict_order(bot(), i) && order(i, top()));
    }
    else if constexpr(sign == SPOS) {
      assert(order(bot(), i) && strict_order(i, top()));
    }
  }
};

template<class ZUniverse>
class ZTotalOrder;

/** Lattice of increasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \leq y\} \f$. */
template<class VT>
using ZInc = ZTotalOrder<ZIncUniverse<VT>>;

/** Lattice of decreasing integers.
Concretization function: \f$ \gamma(x) = \{_ \mapsto y \;|\; x \geq y\} \f$. */
template<class VT>
using ZDec = ZTotalOrder<ZDecUniverse<VT>>;

/** Lattice of increasing positive integer numbers (0 is included) \f$ \mathbb{Z}^+ \f$ (aka. natural numbers \f$ \mathbb{N} \f$).
The concretization is the same than for `ZInc`. */
template<class VT>
using ZPInc = ZTotalOrder<ZIncUniverse<VT, SPOS>>;

/** Lattice of decreasing positive integer numbers (0 is included) \f$ \mathbb{Z}^+ \f$ (aka. natural numbers \f$ \mathbb{N} \f$).
The concretization is the same than for `ZDec`. */
template<class VT>
using ZPDec = ZTotalOrder<ZDecUniverse<VT, SPOS>>;

/** Lattice of increasing negative integer numbers (0 is included) \f$ \mathbb{Z}^- \f$.
The concretization is the same than for `ZInc`. */
template<class VT>
using ZNInc = ZTotalOrder<ZIncUniverse<VT, SNEG>>;

/** Lattice of decreasing negative integer numbers (0 is included) \f$ \mathbb{Z}^- \f$.
The concretization is the same than for `ZDec`. */
template<class VT>
using ZNDec = ZTotalOrder<ZDecUniverse<VT, SNEG>>;

/** Lattice of increasing Boolean where \f$ \mathit{false} \leq \mathit{true} \f$. */
using BInc = ZTotalOrder<ZIncUniverse<bool, BOUNDED>>;

/** Lattice of decreasing Boolean where \f$ \mathit{true} \leq \mathit{false} \f$. */
using BDec = ZTotalOrder<ZDecUniverse<bool, BOUNDED>>;

template<class L, class K> struct join_t;
template<class L, class K> struct meet_t;

template<class TotalOrder>
struct guarded_if {};

template <template <class> class T>   // This template shenanigans because specializing with `BInc` leads to an incomplete type error...
struct guarded_if<T<ZIncUniverse<bool, BOUNDED>>> {
  CUDA bool guard() const { return static_cast<const T<ZIncUniverse<bool, BOUNDED>>&>(*this).value(); }
};

template<class ZUniverse>
class ZTotalOrder : public guarded_if<ZTotalOrder<ZUniverse>> {
public:
  constexpr static bool increasing = ZUniverse::increasing;
  constexpr static bool decreasing = ZUniverse::decreasing;
  using ValueType = typename ZUniverse::ValueType;
  using this_type = ZTotalOrder<ZUniverse>;
  using dual_type = ZTotalOrder<typename ZUniverse::dual_type>;

  template<class ZU>
  friend class ZTotalOrder;

  template<class L, class K>
  friend typename join_t<L, K>::type join(L a, K b);

  template<class L, class K>
  friend typename meet_t<L, K>::type meet(L a, K b);

  using U = ZUniverse;

  template<typename T, typename U>
  using IsConvertible = std::enable_if_t<std::is_convertible_v<T, U>, bool>;
private:

  struct no_check_t{};

  ValueType val;

  template<typename VT2, IsConvertible<VT2, ValueType> = true>
  CUDA explicit ZTotalOrder(VT2 i, no_check_t): val(static_cast<ValueType>(i)) {}

public:
  /** Similar to \f$[\![\mathit{true}]\!]\f$. */
  CUDA static this_type bot() {
    return this_type(U::bot(), no_check_t{});
  }

  /** Similar to \f$[\![\mathit{false}]\!]\f$. */
  CUDA static this_type top() {
    return this_type(U::top(), no_check_t{});
  }

  CUDA dual_type dual() const {
    return dual_type(val, typename dual_type::no_check_t{});
  }

  /** Similar to \f$[\![x \geq_A i]\!]\f$ for any name `x` where \f$ \geq_A \f$ is the lattice order. */
  template<typename VT2, IsConvertible<VT2, ValueType> = true>
  CUDA ZTotalOrder(VT2 i): val(static_cast<ValueType>(i)) {
    ZUniverse::check(i);
  }

  CUDA ZTotalOrder(const this_type& other): val(other.val) {}
  CUDA ZTotalOrder(this_type&& other): val(std::move(other.val)) {}
  CUDA this_type& operator=(this_type&& other) {
    ValueType old = std::move(val);
    val = std::move(other.val);
    other.val = std::move(old);
    return *this;
  }

  CUDA const ValueType& value() const { return val; }

  /** Expects a predicate of the form `x <op> i` where `x` is any variable's name, and `i` an integer.
    - If `f.approx()` is EXACT: `op` can be `U::sig_order()` or `U::sig_strict_order()`.
    - If `f.approx()` is UNDER: `op` can be, in addition to exact, `!=`.
    - If `f.approx()` is OVER: `op` can be, in addition to exact, `==`.
    Existential formula \f$ \exists{x:T} \f$ can also be interpreted (only to bottom).
    - The type `Int` is supported regardless of the approximation.
    - If `f.approx()` is UNDER, then `T` can also be equal to `Real`.
    */
  template<typename Formula>
  CUDA static thrust::optional<this_type> interpret(const Formula& f) {
    if(f.is_true()) {
      return bot();
    }
    else if(f.is_false()) {
      return top();
    }
    else if(f.is(Formula::E)) {
      CType ty = get<1>(f.exists());
      if(ty == Int) {
        return bot();
      }
      else if(ty == Real && f.approx() == UNDER) {
        return bot();
      }
    }
    else if(is_v_op_z(f, U::sig_order())) {      // e.g., x <= 4
      return this_type(f.seq(1).z());
    }
    else if(is_v_op_z(f, U::sig_strict_order())) {  // e.g., x < 4
      return this_type(U::next(f.seq(1).z()));
    }
    // Under-approximation of `x != 4` as `next(4)`.
    else if(is_v_op_z(f, NEQ) && f.approx() == UNDER) {
      return this_type(U::next(f.seq(1).z()));
    }
    // Over-approximation of `x == 4` as `4`.
    else if(is_v_op_z(f, EQ) && f.approx() == OVER) {
      return this_type(f.seq(1).z());
    }
    return {};
  }

  /** `true` whenever \f$ a = \top \f$, `false` otherwise. */
  CUDA BInc is_top() const {
    return val == U::top();
  }

  /** `true` whenever \f$ a = \bot \f$, `false` otherwise. */
  CUDA BDec is_bot() const {
    return val == U::bot();
  }

  CUDA this_type& tell(const this_type& other, BInc& has_changed) {
    if(U::strict_order(this->val, other.val)) {
      this->val = other.val;
      has_changed.val = true;
    }
    return *this;
  }

  CUDA this_type& dtell(const this_type& other, BInc& has_changed) {
    if(U::strict_order(other.val, this->val)) {
      this->val = other.val;
      has_changed.val = true;
    }
    return *this;
  }

  /** \return \f$ x \geq i \f$ where `x` is a variable's name and `i` the integer value.
  `true` is returned whenever \f$ a = \bot \f$ and `false` whenever \f$ a = \top \f$. */
  template<class Allocator>
  CUDA TFormula<Allocator> deinterpret(const LVar<Allocator>& x, const Allocator& allocator = Allocator()) const {
    if(is_top().value()) {
      return TFormula<Allocator>::make_false();
    }
    else if(is_bot().value()) {
      return TFormula<Allocator>::make_true();
    }
    return make_v_op_z(x, U::sig_order(), val, EXACT, allocator);
  }

  template<class Allocator>
  CUDA DArray<this_type, Allocator> split(const Allocator& allocator = Allocator()) const {
    if(is_top().guard()) {
      return DArray<this_type, Allocator>();
    }
    else {
      return DArray<this_type, Allocator>(1, *this, allocator);
    }
  }

  /** \return A copy of the current abstract element. */
  CUDA this_type clone() const { return *this; }

  /** Print the current element. */
  CUDA void print() const {
    if(is_bot().value()) {
      printf("%c", 0x22A5);
    }
    else if(is_top().value()) {
      printf("%c", 0x22A4);
    }
    else if(val >= 0) {
      printf("%llu", (unsigned long long int) val);
    }
    else {
      printf("%lld", (long long int) val);
    }
  }
};


template<class U>
class spos {
  private:
    U v;
  public:
    using ValueType = U;
    spos(U v): v(v) { assert(v >= 0); }
    operator U() const { return v; }
    U value() const { return v; }
};

template<class U>
class sneg {
  private:
    U v;
  public:
    using ValueType = U;
    sneg(U v): v(v) { assert(v <= 0); }
    operator U() const { return v; }
    U value() const { return v; }
};

#include "monotone_analysis.hpp"

template<class L, class K>
CUDA typename join_t<L, K>::type join(L a, K b) {
  using R = typename join_t<L, K>::type;
  return R(R::U::join(unwrap(a), unwrap(b)), typename R::no_check_t{});
}

template<class L, class K>
CUDA typename meet_t<L, K>::type meet(L a, K b) {
  using R = typename meet_t<L, K>::type;
  return R(R::U::meet(unwrap(a), unwrap(b)), typename R::no_check_t{});
}

template<class O, class L, class K>
CUDA typename leq_t<O, L, K>::type leq(L a, K b) {
  using R = typename leq_t<O, L, K>::type;
  return R(O::U::order(unwrap(a), unwrap(b)));
}

template<class O, class L, class K>
CUDA typename lt_t<O, L, K>::type lt(L a, K b) {
  using R = typename lt_t<O, L, K>::type;
  return R(O::U::strict_order(unwrap(a), unwrap(b)));
}

} // namespace lala

#endif
