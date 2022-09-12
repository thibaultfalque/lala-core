// Copyright 2021 Pierre Talbot

#ifndef IRESULT_HPP
#define IRESULT_HPP

#include "utility.hpp"
#include "vector.hpp"
#include "string.hpp"
#include "string.hpp"
#include "tuple.hpp"
#include "variant.hpp"
#include "logic/ast.hpp"

namespace lala {

/** The representation of an error (or warning) obtained when interpreting a formula in an abstract universe or domain. */
template<class F>
class IError {
public:
  using allocator_type = typename F::allocator_type;
  using this_type = IError<F>;

private:
  battery::string<allocator_type> ad_name;
  battery::string<allocator_type> description;
  F uninterpretable_formula;
  AType aty;
  battery::vector<IError<F>, allocator_type> suberrors;

  CUDA void print_indent(int indent) {
    for(int i = 0; i < indent; ++i) {
      printf(" ");
    }
  }

  CUDA void print_line(const char* line, int indent) {
    print_indent(indent);
    printf(line);
  }

public:
  // If fatal is false, it is considered as a warning.
  CUDA IError(bool fatal, battery::string<allocator_type> ad_name,
    battery::string<allocator_type> description,
    F uninterpretable_formula,
    AType aty = UNTYPED)
   : ad_name(std::move(ad_name)),
     description(std::move(description)),
     uninterpretable_formula(std::move(uninterpretable_formula)),
     aty(aty)
  {}

  CUDA this_type& add_suberror(IError<F>&& suberror) {
    suberrors.push_back(suberror);
    return *this;
  }

  CUDA void print(int indent = 0) const {
    if(fatal) {
      print_line("[error] ", indent);
    }
    else {
      print_line("[warning] ", indent);
    }
    printf("Uninterpretable formula.\n");
    print_indent(indent);
    printf("  Abstract domain: %s\n", ad_name.data());
    print_line("  Abstract type: ", indent);
    if(aty == UNTYPED) {
      printf("untyped\n");
    }
    else {
      printf("%d\n", aty);
    }
    print_line("  Formula: ");
    uninterpretable_formula.print(true);
    printf("\n");
    print_indent(indent);
    printf("  Description: %s\n", description.data());
    for(int i = 0; i < suberrors.size(); ++i) {
      suberrors[i].print(indent + 2);
      printf("\n");
    }
  }
}

/** This class is used in abstract domains to represent the result of an interpretation.
    If the abstract domain cannot interpret the formula, it must explain why.
    This is similar to compilation errors in compiler. */
template <class T, class F>
class IResult {
public:
  using allocator_type = typename F::allocator_type;
  using error_type = IError<F>;

private:
  using warnings_type = battery::vector<error_type, allocator_type>;

  using result_type = battery::variant<
    T,
    error_type>;

  result_type result;
  warnings_type warnings;

  template <class U>
  CUDA static result_type map_result(battery::variant<U, error_type>&& other) {
    if(other.index() == 0) {
      return result_type::template create<0>(T(std::move(battery::get<0>(other))));
    }
    else {
      return result_type::template create<1>(std::move(battery::get<1>(other)));
    }
  }

public:
  CUDA IResult(T&& data):
    result(result_type::template create<0>(std::move(data))) {}

  CUDA this_type& push_warning(error_type&& warning) {
    warnings.push_back(std::move(warning));
  }

  CUDA IResult(error_type&& error):
    result(result_type::template create<1>(std::move(error))) {}

  template<class U>
  CUDA IResult(IResult<U, F>&& map): result(map_result(std::move(map.result))),
    warnings(std::move(map.warnings)) {}

  CUDA bool is_ok() const {
    return result.index() == 0;
  }

  CUDA const T& value() const {
    return battery::get<0>(result);
  }

  template<class U>
  CUDA IResult<U, F> map(U&& data2) && {
    auto r = IResult<U, F>(data2);
    r.warnings = std::move(warnings);
    return r;
  }

  CUDA T& value() {
    return battery::get<0>(result);
  }

  CUDA const error_type& error() const {
    return battery::get<1>(result);
  }

  // CUDA this_type& map_value(T&& new_data) {
  //   data = new_data;
  //   return *this;
  // }

  CUDA void print_diagnostics() const {
    if(is_ok()) {
      printf("successfully interpreted\n");
    }
    else {
      error().print();
    }
    printf("\n");
    for(int i = 0; i < warnings.size(); ++i) {
      warnings[i].print();
    }
  }
};

}

#endif