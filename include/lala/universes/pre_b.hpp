// Copyright 2022 Pierre Talbot

#ifndef LALA_CORE_PRE_B_HPP
#define LALA_CORE_PRE_B_HPP

#include "../logic/logic.hpp"

namespace lala {

struct PreBD;

/** `PreB` is a domain abstracting the Boolean universe of discourse \f$ \mathbb{B}=\{true,false\} \f$.
    We have \f$ PreB \triangleq \langle \{f,t\}, \implies, \lor, \land, f, t \rangle \f$ where the order, join and meet operations are given by the usual Boolean logical connectors.
    Picturally, we have f as the bottom element and t as the top element.
    As shown by the Galois connection below, the element `t` does not represent the truth value but rather the unknown value (either true or false).
    Interpreting `t` as true to implement the lattice operations is only a matter of convenience since those are the same than the logical operations when viewing `t` as true.
    It is also useful to obtain the dual lattice \f$ PreBD \triangleq \langle \{f,t\}, \Leftarrow, \land, \lor, t, f \rangle \f$ by reusing already implemented operations.
    However, it becomes different when we consider functions (`fun<sig>`) over this domain, as those might be different in `PreB` and `PreBD`.
    For the implication, we have \f$ x \implies y \triangleq t \f$ in PreB for any values of x and y, but we have \f$ x \implies y \triangleq (x = t /\ y = t) ? t : f \f$ in PreBD, hence both cases need to be handled separately and no duality can be used (the function \f$ \implies \f$ is computed w.r.t. the concrete domain, which is not the same than the order of this lattice).

    We have a Galois connection between the concrete domain of values \f$ \mathcal{P}(\mathbb{B}) \f$ and PreB:
    * Concretization: \f$ \gamma(b) \triangleq b = t ? \{\mathit{true}, \mathit{false}\} : \{\mathit{false}\} \f$.
    * Abstraction: \f$ \alpha(S) \triangleq \mathit{true} \in S ? t : f \f$.

    Note that by taking this lattice and its dual (using `Interval<Bound<PreB>>`), you can obtain Dunn/Belnap logic with a knowledge ordering.
*/
struct PreB {
  using this_type = PreB;
  using dual_type = PreBD;
  using value_type = bool;

  constexpr static const bool is_natural = true; /** We consider \f$ \top = \mathit{true} \f$ is the natural order on Boolean. */
  using natural_order = PreB;

  constexpr static const bool is_totally_ordered = true;
  constexpr static const bool preserve_bot = false; /** \f$ \gamma(\mathit{false}) = \{\mathit{false}\} \f$, therefore the empty set cannot be represented in this domain. */
  constexpr static const bool preserve_top = true; /** \f$ \gamma(\mathit{unknown}) = \{false, true\} \f$ */
  constexpr static const bool preserve_join = true; /** \f$ \gamma(x \sqcup y) = \gamma(x) \cup \gamma(y) \f$ */
  constexpr static const bool preserve_meet = true; /** \f$ \gamma(x \sqcap y) = \gamma(x) \cap \gamma(y) \f$ */
  constexpr static const bool injective_concretization = true; /** Each element of PreB maps to a different concrete value. */
  constexpr static const bool preserve_concrete_covers = true; /** \f$ x \lessdot y \Leftrightarrow \gamma(x) \lessdot \gamma(y) \f$ */
  constexpr static const char* name = "B";
  constexpr static const bool is_arithmetic = true;
  CUDA constexpr static value_type zero() { return false; }
  CUDA constexpr static value_type one() { return true; }

  /** @sequential
   * Interpret a formula \f$ x => \mathit{false} \f$ into the PreB lattice.
   * \return `true` if `f` is the false constant. Otherwise it returns `false` with a diagnostic. */
  template<bool diagnose, class F, bool dualize=false>
  CUDA static bool interpret_tell(const F& f, value_type& tell, IDiagnostics& diagnostics) {
    if(f.is(F::B)) {
      if constexpr(dualize) {
        if(!f.b()) {
          INTERPRETATION_ERROR("The constant `false` would be overapproximated by the top element (which concretization gives {true, false}) in the `PreBD` domain.");
        }
      }
      else {
        if(f.b()) {
          INTERPRETATION_ERROR("The constant `true` would be overapproximated by the top element (which concretization gives {true, false}) in the `PreB` domain.");
        }
      }
      tell = f.b();
      return true;
    }
    RETURN_INTERPRETATION_ERROR("Only constant of types `Bool` can be interpreted in a Boolean domain.");
  }

  /** @sequential
   * We can only ask \f$ x => \mathit{false} \f$ if an element of this lattice is `false`, because it cannot exactly represent `true`.
   * This operation can be dualized.
  */
  template<bool diagnose, class F, bool dualize=false>
  CUDA static bool interpret_ask(const F& f, value_type& ask, IDiagnostics& diagnostics) {
    return interpret_tell<diagnose, F, dualize>(f, ask, diagnostics);
  }

  /** @sequential
   * Verify if the type of a variable, introduced by an existential quantifier, is compatible with the current abstract universe.
   * \return `bot()` if the type of the existentially quantified variable is `Bool`. Otherwise it returns an explanation of the error.
   * This operation can be dualized.
  */
  template<bool diagnose, class F, bool dualize = false>
  CUDA NI static bool interpret_type(const F& f, value_type& k, IDiagnostics& diagnostics) {
    assert(f.is(F::E));
    const auto& cty = battery::get<1>(f.exists());
    if(cty.is_bool()) {
      k = dualize ? top() : bot();
      return true;
    }
    else {
      const auto& vname = battery::get<0>(f.exists());
      RETURN_INTERPRETATION_ERROR("The type of `" + vname + "` can only be `Bool` when interpreted in Boolean domains.");
    }
  }

  /** @parallel
   * Given a Boolean value, create a logical constant representing that value.
  */
  template<class F>
  CUDA static F formula_of_constant(const value_type& v) {
    return F::make_bool(v);
  }

  /** The logical predicate symbol corresponding to the order of this pre-universe.
      \return The logical symbol `IMPLY`. */
  CUDA static constexpr Sig sig_order() { return IMPLY; }

  /** Converse nonimplication: we have a < b only when `a` is `false` and `b` is `true`. */
  CUDA static constexpr Sig sig_strict_order() { return LT; }

  /** \f$ \bot \f$ is represented by `false`. */
  CUDA static constexpr value_type bot() { return false; }

  /** \f$ \top \f$ is represented by `true`. */
  CUDA static constexpr value_type top() { return true; }

  /** \return \f$ x \sqcup y \f$ defined as \f$ x \lor y \f$. */
  CUDA static constexpr value_type join(value_type x, value_type y) { return x || y; }

  /** \return \f$ x \sqcap y \f$ defined as \f$ x \land y \f$. */
  CUDA static constexpr value_type meet(value_type x, value_type y) { return x && y; }

  /** \return \f$ \mathit{true} \f$ if \f$ x \implies y \f$. */
  CUDA static constexpr bool order(value_type x, value_type y) { return !x || y; }

/** \return \f$ \mathit{true} \f$ if \f$ x \nleftarrow y \f$ where the converse nonimplication is true only when \f$ \mathit{false} \nleftarrow \mathit{true} \f$, otherwise it is false. */
  CUDA static constexpr bool strict_order(value_type x, value_type y) { return !x && y; }

  /**  From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ y \f$ is a cover of \f$ x \f$, or top if x is top.

    \return `true`. */
  CUDA static constexpr value_type next(value_type x) { return true; }

  /** From a lattice perspective, this function returns an element \f$ y \f$ such that \f$ x \f$ is a cover of \f$ y \f$, or bot if x is bot.

   \return `false`. */
  CUDA static constexpr value_type prev(value_type x) { return false; }

  /** We do not support function that trivially maps to the \f$ \top \f$ element.
   * For instance, the logical negation `not` is not supported because it is defined by: \f$ \lnot x \triangleq x = \mathit{true} \f$, because the negation of false is overapproximated by \f$ \top \f$ representing \f$ {true,false} \f$ in the concrete domain, and the negation of true stays true ( \f$ \lnot {true,false} = {\lnot true, \lnot false} = {true,false} \f$).
   */
  CUDA NI static constexpr bool is_supported_fun(Sig sig) {
    switch(sig) {
      case AND:
      case OR:
      case IMPLY:
      case EQUIV:
      case XOR:
      case EQ:
      case NEQ:
        return true;
      case NOT:
      default:
        return false;
    }
  }

  CUDA NI static constexpr bool is_order_preserving(Sig sig) {
    switch(sig) {
      case AND:
      case OR:
      case IMPLY:
      case EQUIV:
      case XOR:
      case EQ:
      case NEQ:
        return true;
      default:
        return false;
    }
  }

  /** We support binary logical connectors.
   *  * \f$ \lnot x \triangleq x = \mathit{true} \f$, simply because the negation of false is overapproximated by \f$ \top \f$ representing \f$ {true,false} \f$ in the concrete domain, and the negation of true stays true ( \f$ \lnot {true,false} = {\lnot true, \lnot false} = {true,false} \f$).
   */
  template<Sig sig, bool dualize=false>
  CUDA NI static constexpr value_type fun(value_type x, value_type y) {
    static_assert(sig == AND || sig == OR || sig == IMPLY || sig == EQUIV || sig == XOR,
      "Unsupported binary function.");
    switch(sig) {
      case AND: return dualize ? x || y : x && y;
      case OR: return dualize ? x && y : x || y;
      case IMPLY: return dualize ? !y || x : !x || y;
      case EQUIV:
      case EQ: return x == y;
      case XOR:
      case NEQ: return x != y;
      default: assert(0); return x;
    }
  }
};

} // namespace lala

#endif
