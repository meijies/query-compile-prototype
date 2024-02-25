#[cfg(test)]
mod tests {
    use egg::{rewrite as rw, AstSize, EGraph, Extractor, Pattern, RecExpr, Rewrite, Runner, Searcher, SymbolLang};

    #[test]
    fn test_basic() {
        let my_expr: RecExpr<SymbolLang> = "(foo a b)".parse().unwrap();
        println!(" this is my expression {}", my_expr);

        let mut expr = RecExpr::default();
        let a = expr.add(SymbolLang::leaf("a"));
        let b = expr.add(SymbolLang::leaf("b"));
        let foo = expr.add(SymbolLang::new("foo", vec![a, b]));

        let mut egraph: EGraph<SymbolLang, ()> = Default::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let foo = egraph.add(SymbolLang::new("foo", vec![a, b]));

        let foo2 = egraph.add_expr(&expr);
        assert_eq!(foo, foo2);
    }

    #[test]
    fn test_search_rewrite() {
        let mut egraph: EGraph<SymbolLang, ()> = Default::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let foo = egraph.add(SymbolLang::new("foo", vec![a, b]));

        egraph.rebuild();

        let pat: Pattern<SymbolLang> = "(foo ?x ?x)".parse().unwrap();
        let matches = pat.search(&egraph);
        assert!(matches.is_empty());

        egraph.union(a, b);
        egraph.rebuild();

        let matches = pat.search(&egraph);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_runner() {
        let rules: &[Rewrite<SymbolLang, ()>] = &[
            rw!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"),
            rw!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"),
            rw!("add-0"; "(+ ?x 0)" => "(?x)"),
            rw!("mul-0"; "(* ?x 0)" => "0"),
            rw!("mul-1"; "(* ?x 1)" => "?x"),
        ];

        let start = "(+ 0 (* 1 a))".parse().unwrap();
        let runner = Runner::default().with_expr(&start).run(rules);

        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);

        assert_eq!(best_expr, "a".parse().unwrap());
        assert_eq!(best_cost, 1);

    }
}
