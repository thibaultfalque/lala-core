// Copyright 2021 Pierre Talbot

#ifndef ENV_HPP
#define ENV_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "string.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "logic/ast.hpp"
#include "logic/iresult.hpp"

namespace lala {

/** A `VarEnv` is a variable environment mapping between logical variables and abstract variables. */
template <class Allocator>
class VarEnv {
public:
  using allocator_type = Allocator;
  using this_type = VarEnv<Allocator>;

  template<class F>
  using iresult = IResult<AVar, F>;

  constexpr static const char* name = "VarEnv";

  template<class T>
  using bvector = battery::vector<T, Allocator>;
  using bstring = battery::string<Allocator>;

  struct Variable {
    bstring name;
    Sort<Allocator> sort;
    Approx appx;
    bvector<AVar> avars;
    CUDA Variable(const bstring& name, const Sort<Allocator>& sort, Approx appx, AVar av)
      : name(name), sort(sort), appx(appx)
    {
      avars.push_back(av);
    }

    CUDA thrust::optional<AVar> avar_of(AType aty) const {
      for(int i = 0; i < avars.size(); ++i) {
        if(AID(avars[i]) == aty) {
          return avars[i];
        }
      }
      return {};
    }
  };
private:
  bvector<Variable> lvars;
  bvector<bvector<int>> avar2lvar;

  CUDA void extends_abstract_doms(AType aty) {
    assert(aty != UNTYPED);
    while(aty >= avar2lvar.size()) {
      avar2lvar.push_back(bvector<int>());
    }
  }

  CUDA AVar extends_vars(AType aty, const bstring& name, const Sort<Allocator>& sort, Approx appx) {
    extends_abstract_doms(aty);
    AVar avar = make_var(aty, avar2lvar[aty].size());
    avar2lvar[aty].push_back(lvars.size());
    lvars.push_back(Variable(name, sort, appx, avar));
    return avar;
  }

  template <class F>
  CUDA iresult<F> interpret_existential(const F& f) {
    const auto& vname = battery::get<0>(f.exists());
    if(f.type() == UNTYPED) {
      return iresult<F>(IError<F>(true, name, "Untyped abstract type: variable `" + vname + "` has no abstract type.", f));
    }
    if(contains(vname)) {
      return iresult<F>(IError<F>(true, name, "Invalid redeclaration: variable `" + vname + "` has already been declared.", f));
    }
    else {
      AType aty = f.type();
      const Sort<Allocator>& sort = battery::get<1>(f.exists());
      return iresult<F>(extends_vars(aty, vname, sort, f.approx()));
    }
  }

  template <class F>
  CUDA iresult<F> interpret_lv(const F& f) {
    const auto& vname = f.lv();
    auto var = variable_of(vname);
    if(var.has_value()) {
      if(f.type() != UNTYPED) {
        auto avar = var->avar_of(f.type());
        if(avar.has_value()) {
          return AVar(*avar);
        }
        else {
          return iresult<F>(IError<F>(true, name, "Variable `" + vname + "` has not been declared in the abstract domain `" + bstring::from_int(f.type()) + "`.", f));
        }
      }
      else {
        if(var->avars.size() == 1) {
          return AVar(var->avars[0]);
        }
        else {
          return iresult<F>(IError<F>(true, name, "Variable occurrence `" + vname + "` is untyped, but exists in multiple abstract domains.", f));
        }
      }
    }
    else {
      return iresult<F>(IError<F>(true, name, "Undeclared variable `" + vname + "`.", f));
    }
  }

public:
  CUDA VarEnv(const Allocator& allocator): lvars(allocator), avar2lvar(allocator) {}
  CUDA VarEnv(): VarEnv(Allocator()) {}

  CUDA allocator_type get_allocator() const {
    return lvars.get_allocator();
  }

  CUDA size_t num_abstract_doms() const {
    return avar2lvar.size();
  }

  CUDA size_t num_vars() const {
    return lvars.size();
  }

  CUDA size_t num_vars_in(AType aty) const {
    return avar2lvar[aty].size();
  }

  /** A variable environment can interpret formulas of two forms:
   *    - Existential formula with a valid abstract type (`f.type() != UNTYPED`).
   *    - Variable occurrence.
   * It returns an abstract variable (`AVar`) corresponding to the variable created (existential) or already presents (occurrence). */
  template <class F>
  CUDA iresult<F> interpret(const F& f) {
    if(f.is(F::E)) {
      return interpret_existential(f);
    }
    else if(f.is(F::LV)) {
      return interpret_lv(f);
    }
    else if(f.is(F::V)) {
      if(contains(f.v())) {
        return f.v();
      }
      else {
        return iresult<F>(IError<F>(true, name, "Undeclared abstract variable `" + bstring::from_int(f.v()) + "`.", f));
      }
    }
    else {
      return iresult<F>(IError<F>(true, name, "Unsupported formula: `VarEnv` can only interpret quantifiers and occurrences of variables.", f));
    }
  }

  CUDA thrust::optional<const Variable&> variable_of(const bstring& lv) const {
    for(int i = 0; i < lvars.size(); ++i) {
      if(lvars[i].name == lv) {
        return lvars[i];
      }
    }
    return {};
  }

  CUDA bool contains(const bstring& lv) const {
    return variable_of(lv).has_value();
  }

  CUDA bool contains(AVar av) const {
    if(av != UNTYPED) {
      return avar2lvar.size() > AID(av) && avar2lvar[AID(av)].size() > VID(av);
    }
    return false;
  }

  CUDA const Variable& operator[](AVar av) const {
    return lvars[avar2lvar[AID(av)][VID(av)]];
  }

  CUDA const bstring& name_of(AVar av) const {
    return (*this)[av].name;
  }

  CUDA const Sort<Allocator>& sort_of(AVar av) const {
    return (*this)[av].sort;
  }

  CUDA Approx approx_of(AVar av) const {
    return (*this)[av].appx;
  }
};

/** Given a formula `f` and an environment, return the first abstract variable (`AVar`) occurring in `f` or `{}` if `f` has no variable in `env`. */
template <class F, class Env>
CUDA thrust::optional<const typename Env::Variable&> var_in(const F& f, const Env& env) {
  const auto& g = var_in(f);
  switch(g.index()) {
    case F::V:
      return env[g.v()];
    case F::E:
      return env.variable_of(battery::get<0>(g.exists()));
    case F::LV:
      return env.variable_of(g.lv());
  }
  return {};
}
}

#endif
