use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use good_lp::{
    highs, variable, Expression, ProblemVariables, Solution, SolutionStatus, SolverModel,
    WithTimeLimit,
};

pub const PSEUDO_ROOT_ID: usize = 0;
pub const PSEUDO_ROOT_COLOR: i32 = -1;

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentVertex {
    pub formula: String,
    pub counts: Vec<i32>,
    pub ionization: String,
    pub peak_id: Option<usize>,
    pub color: i32,
    pub mass: f64,
}

impl FragmentVertex {
    pub fn pseudo_root(n_elements: usize) -> Self {
        Self {
            formula: String::new(),
            counts: vec![0; n_elements],
            ionization: String::new(),
            peak_id: None,
            color: PSEUDO_ROOT_COLOR,
            mass: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LossEdge {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
    pub artificial: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentationGraph {
    pub fragments: Vec<FragmentVertex>,
    pub edges: Vec<LossEdge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentCandidate {
    pub formula: String,
    pub counts: Vec<i32>,
    pub ionization: String,
    pub peak_id: usize,
    pub color: i32,
    pub mass: f64,
    pub score: f64,
}

impl FragmentCandidate {
    fn as_vertex(&self) -> FragmentVertex {
        FragmentVertex {
            formula: self.formula.clone(),
            counts: self.counts.clone(),
            ionization: self.ionization.clone(),
            peak_id: Some(self.peak_id),
            color: self.color,
            mass: self.mass,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubFormulaGraphInput {
    pub root_candidates: Vec<FragmentCandidate>,
    pub fragment_candidates: Vec<FragmentCandidate>,
    pub allowed_ionizations: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphScoring {
    pub peak_scores: BTreeMap<i32, f64>,
    pub peak_pair_scores: BTreeMap<(i32, i32), f64>,
    pub fragment_scores: BTreeMap<String, f64>,
    pub loss_scores: BTreeMap<(String, String), f64>,
    pub general_graph_score: f64,
}

impl GraphScoring {
    pub fn scored_root_weight(&self, root: &FragmentCandidate) -> f64 {
        root.score
            + score_for_color(&self.peak_scores, root.color)
            + score_for_formula(&self.fragment_scores, &root.formula)
            + self.general_graph_score
    }

    pub fn scored_loss_weight(
        &self,
        parent: &FragmentVertex,
        child: &FragmentCandidate,
        artificial: bool,
    ) -> f64 {
        let mut score = child.score
            + score_for_color(&self.peak_scores, child.color)
            + score_for_formula(&self.fragment_scores, &child.formula);
        if parent.color != PSEUDO_ROOT_COLOR && !artificial {
            score += self
                .peak_pair_scores
                .get(&(parent.color, child.color))
                .copied()
                .unwrap_or(0.0);
            score += self
                .loss_scores
                .get(&(parent.formula.clone(), child.formula.clone()))
                .copied()
                .unwrap_or(0.0);
        }
        score
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentationTreeComputation {
    pub graph: FragmentationGraph,
    pub reduced_graph: Option<FragmentationGraph>,
    pub tree: FragmentationTree,
}

impl FragmentationGraph {
    pub fn new(n_elements: usize) -> Self {
        Self {
            fragments: vec![FragmentVertex::pseudo_root(n_elements)],
            edges: Vec::new(),
        }
    }

    pub fn add_fragment(&mut self, fragment: FragmentVertex) -> usize {
        let id = self.fragments.len();
        self.fragments.push(fragment);
        id
    }

    pub fn add_loss(
        &mut self,
        source: usize,
        target: usize,
        weight: f64,
        artificial: bool,
    ) -> usize {
        let id = self.edges.len();
        self.edges.push(LossEdge {
            source,
            target,
            weight,
            artificial,
        });
        id
    }

    pub fn incoming_edges(&self) -> Vec<Vec<usize>> {
        let mut incoming = vec![Vec::new(); self.fragments.len()];
        for (edge_id, edge) in self.edges.iter().enumerate() {
            if edge.target < incoming.len() {
                incoming[edge.target].push(edge_id);
            }
        }
        incoming
    }

    pub fn outgoing_edges(&self) -> Vec<Vec<usize>> {
        let mut outgoing = vec![Vec::new(); self.fragments.len()];
        for (edge_id, edge) in self.edges.iter().enumerate() {
            if edge.source < outgoing.len() {
                outgoing[edge.source].push(edge_id);
            }
        }
        outgoing
    }

    pub fn validate(&self) -> Result<(), TreeSolveError> {
        if self.fragments.is_empty() {
            return Err(TreeSolveError::InvalidGraph(
                "fragmentation graph must contain the pseudo-root".to_string(),
            ));
        }
        if self.fragments[PSEUDO_ROOT_ID].color != PSEUDO_ROOT_COLOR {
            return Err(TreeSolveError::InvalidGraph(
                "fragment 0 must be the pseudo-root with color -1".to_string(),
            ));
        }
        for (idx, edge) in self.edges.iter().enumerate() {
            if edge.source >= self.fragments.len() || edge.target >= self.fragments.len() {
                return Err(TreeSolveError::InvalidGraph(format!(
                    "edge {idx} references an out-of-range fragment"
                )));
            }
            if edge.target == PSEUDO_ROOT_ID {
                return Err(TreeSolveError::InvalidGraph(format!(
                    "edge {idx} points into the pseudo-root"
                )));
            }
            if !edge.weight.is_finite() {
                return Err(TreeSolveError::InvalidGraph(format!(
                    "edge {idx} has non-finite weight"
                )));
            }
        }
        if self.edges.iter().all(|edge| edge.source != PSEUDO_ROOT_ID) {
            return Err(TreeSolveError::InvalidGraph(
                "fragmentation graph has no pseudo-root outgoing edge".to_string(),
            ));
        }
        topological_order(self)?;
        Ok(())
    }

    pub fn simple_reduction(&self) -> Result<Self, TreeSolveError> {
        let mut reduced = self.remove_nonfinite_edges().remove_unreachable();

        loop {
            let upper_bounds = optimistic_upper_bounds(&reduced)?;
            let before = reduced.edges.len();
            reduced.edges.retain(|edge| {
                edge.source == PSEUDO_ROOT_ID || edge.weight + upper_bounds[edge.target] >= 0.0
            });
            reduced = reduced.remove_unreachable();
            if reduced.edges.len() == before {
                return Ok(reduced);
            }
        }
    }

    fn remove_nonfinite_edges(&self) -> Self {
        let mut graph = self.clone();
        graph.edges.retain(|edge| edge.weight.is_finite());
        graph
    }

    fn remove_unreachable(&self) -> Self {
        let outgoing = self.outgoing_edges();
        let mut reachable = vec![false; self.fragments.len()];
        let mut queue = VecDeque::new();
        reachable[PSEUDO_ROOT_ID] = true;
        queue.push_back(PSEUDO_ROOT_ID);

        while let Some(fragment_id) = queue.pop_front() {
            for edge_id in &outgoing[fragment_id] {
                let target = self.edges[*edge_id].target;
                if !reachable[target] {
                    reachable[target] = true;
                    queue.push_back(target);
                }
            }
        }

        let mut old_to_new = vec![usize::MAX; self.fragments.len()];
        let mut fragments = Vec::new();
        for (old_id, fragment) in self.fragments.iter().enumerate() {
            if reachable[old_id] {
                old_to_new[old_id] = fragments.len();
                fragments.push(fragment.clone());
            }
        }

        let edges = self
            .edges
            .iter()
            .filter_map(|edge| {
                if reachable[edge.source] && reachable[edge.target] {
                    Some(LossEdge {
                        source: old_to_new[edge.source],
                        target: old_to_new[edge.target],
                        weight: edge.weight,
                        artificial: edge.artificial,
                    })
                } else {
                    None
                }
            })
            .collect();

        Self { fragments, edges }
    }
}

pub fn build_subformula_graph(
    input: SubFormulaGraphInput,
) -> Result<FragmentationGraph, TreeSolveError> {
    build_subformula_graph_with(input, |_, child| Some(child.score))
}

pub fn build_scored_subformula_graph(
    input: SubFormulaGraphInput,
    scoring: &GraphScoring,
) -> Result<FragmentationGraph, TreeSolveError> {
    build_scored_subformula_graph_with(input, scoring, |_, _| true)
}

pub fn build_scored_subformula_graph_with<F>(
    mut input: SubFormulaGraphInput,
    scoring: &GraphScoring,
    mut is_loss_allowed: F,
) -> Result<FragmentationGraph, TreeSolveError>
where
    F: FnMut(&FragmentVertex, &FragmentCandidate) -> bool,
{
    for root in &mut input.root_candidates {
        root.score = scoring.scored_root_weight(root);
    }
    build_subformula_graph_with(input, |parent, child| {
        if is_loss_allowed(parent, child) {
            Some(scoring.scored_loss_weight(parent, child, false))
        } else {
            None
        }
    })
}

pub fn compute_fragmentation_tree(
    input: SubFormulaGraphInput,
    scoring: &GraphScoring,
    solve_options: TreeSolveOptions,
    reduce_graph: bool,
) -> Result<FragmentationTreeComputation, TreeSolveError> {
    let graph = build_scored_subformula_graph(input, scoring)?;
    compute_fragmentation_tree_from_graph(graph, solve_options, reduce_graph)
}

pub fn compute_fragmentation_tree_from_graph(
    graph: FragmentationGraph,
    solve_options: TreeSolveOptions,
    reduce_graph: bool,
) -> Result<FragmentationTreeComputation, TreeSolveError> {
    let reduced_graph = if reduce_graph {
        Some(graph.simple_reduction()?)
    } else {
        None
    };
    let tree_graph = reduced_graph.as_ref().unwrap_or(&graph);
    let tree = solve_optimal_colorful_tree(tree_graph, solve_options)?;

    Ok(FragmentationTreeComputation {
        graph,
        reduced_graph,
        tree,
    })
}

pub fn build_subformula_graph_with<F>(
    input: SubFormulaGraphInput,
    mut loss_edge_score: F,
) -> Result<FragmentationGraph, TreeSolveError>
where
    F: FnMut(&FragmentVertex, &FragmentCandidate) -> Option<f64>,
{
    let n_elements = validate_graph_build_input(&input)?;
    let allowed_ionizations = input.allowed_ionizations;
    let mut root_candidates = input.root_candidates;
    let mut fragment_candidates = input.fragment_candidates;

    let mut envelope = vec![0_i32; n_elements];
    for root in &root_candidates {
        for (idx, count) in root.counts.iter().enumerate() {
            envelope[idx] = envelope[idx].max(*count);
        }
    }

    let mut graph = FragmentationGraph::new(n_elements);
    for root in root_candidates.drain(..) {
        let root_id = graph.add_fragment(root.as_vertex());
        graph.add_loss(PSEUDO_ROOT_ID, root_id, root.score, false);
    }

    fragment_candidates.sort_by(|left, right| {
        right
            .mass
            .partial_cmp(&left.mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for candidate in &fragment_candidates {
        if !allowed_ionizations.is_empty()
            && !allowed_ionizations
                .iter()
                .any(|ionization| ionization == &candidate.ionization)
        {
            continue;
        }
        if !is_subformula(&envelope, &candidate.counts) {
            continue;
        }

        let mut candidate_vertex_id = None;
        let existing_count = graph.fragments.len();
        for source_id in 1..existing_count {
            let source = &graph.fragments[source_id];
            if source.color == candidate.color || source.mass <= candidate.mass {
                continue;
            }
            if !is_subformula(&source.counts, &candidate.counts) {
                continue;
            }

            let Some(edge_score) = loss_edge_score(source, candidate) else {
                continue;
            };
            if !edge_score.is_finite() {
                continue;
            }

            let target_id = match candidate_vertex_id {
                Some(target_id) => target_id,
                None => {
                    let target_id = graph.add_fragment(candidate.as_vertex());
                    candidate_vertex_id = Some(target_id);
                    target_id
                }
            };
            graph.add_loss(source_id, target_id, edge_score, false);
        }
    }

    Ok(graph)
}

fn validate_graph_build_input(input: &SubFormulaGraphInput) -> Result<usize, TreeSolveError> {
    let Some(first_root) = input.root_candidates.first() else {
        return Err(TreeSolveError::InvalidGraph(
            "at least one root candidate is required".to_string(),
        ));
    };
    let n_elements = first_root.counts.len();
    if n_elements == 0 {
        return Err(TreeSolveError::InvalidGraph(
            "fragment candidates must contain at least one element count".to_string(),
        ));
    }

    for (kind, candidates) in [
        ("root", input.root_candidates.as_slice()),
        ("fragment", input.fragment_candidates.as_slice()),
    ] {
        for (idx, candidate) in candidates.iter().enumerate() {
            if candidate.counts.len() != n_elements {
                return Err(TreeSolveError::InvalidGraph(format!(
                    "{kind} candidate {idx} has {} counts, expected {n_elements}",
                    candidate.counts.len()
                )));
            }
            if candidate.counts.iter().any(|count| *count < 0) {
                return Err(TreeSolveError::InvalidGraph(format!(
                    "{kind} candidate {idx} has a negative element count"
                )));
            }
            if !candidate.mass.is_finite() || !candidate.score.is_finite() {
                return Err(TreeSolveError::InvalidGraph(format!(
                    "{kind} candidate {idx} has non-finite mass or score"
                )));
            }
        }
    }

    Ok(n_elements)
}

fn score_for_color(scores: &BTreeMap<i32, f64>, color: i32) -> f64 {
    scores.get(&color).copied().unwrap_or(0.0)
}

fn score_for_formula(scores: &BTreeMap<String, f64>, formula: &str) -> f64 {
    scores.get(formula).copied().unwrap_or(0.0)
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeSolveOptions {
    pub minimal_score: Option<f64>,
    pub time_limit_seconds: Option<f64>,
    pub threads: Option<u32>,
    pub solver: TreeSolver,
}

impl Default for TreeSolveOptions {
    fn default() -> Self {
        Self {
            minimal_score: None,
            time_limit_seconds: None,
            threads: None,
            solver: TreeSolver::Highs,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeSolver {
    Highs,
    Gurobi,
}

impl TreeSolver {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "highs" => Ok(Self::Highs),
            "gurobi" => Ok(Self::Gurobi),
            _ => Err(format!(
                "unknown fragmentation tree solver {name:?}; expected 'highs' or 'gurobi'"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Highs => "highs",
            Self::Gurobi => "gurobi",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentationTree {
    pub root_fragment: usize,
    pub selected_edges: Vec<usize>,
    pub selected_fragments: Vec<usize>,
    pub tree_weight: f64,
    pub is_optimal: bool,
    pub status: TreeSolutionStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeSolutionStatus {
    Optimal,
    TimeLimit,
    GapLimit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TreeSolveError {
    InvalidGraph(String),
    Infeasible,
    NoSolution(String),
}

impl fmt::Display for TreeSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeSolveError::InvalidGraph(message) => write!(f, "invalid graph: {message}"),
            TreeSolveError::Infeasible => write!(f, "fragmentation tree ILP is infeasible"),
            TreeSolveError::NoSolution(message) => {
                write!(f, "no fragmentation tree solution: {message}")
            }
        }
    }
}

impl std::error::Error for TreeSolveError {}

pub fn solve_optimal_colorful_tree(
    graph: &FragmentationGraph,
    options: TreeSolveOptions,
) -> Result<FragmentationTree, TreeSolveError> {
    graph.validate()?;

    match options.solver {
        TreeSolver::Highs => solve_optimal_colorful_tree_highs(graph, options),
        TreeSolver::Gurobi => solve_optimal_colorful_tree_gurobi(graph, options),
    }
}

fn solve_optimal_colorful_tree_highs(
    graph: &FragmentationGraph,
    options: TreeSolveOptions,
) -> Result<FragmentationTree, TreeSolveError> {
    let incoming = graph.incoming_edges();
    let outgoing = graph.outgoing_edges();
    let root_edges = &outgoing[PSEUDO_ROOT_ID];
    if root_edges.is_empty() {
        return Err(TreeSolveError::InvalidGraph(
            "fragmentation graph has no pseudo-root outgoing edge".to_string(),
        ));
    }

    let mut variables = ProblemVariables::new();
    let edge_vars: Vec<_> = (0..graph.edges.len())
        .map(|edge_id| variables.add(variable().binary().name(format!("e{edge_id}"))))
        .collect();

    let mut objective = Expression::with_capacity(graph.edges.len());
    for (edge, edge_var) in graph.edges.iter().zip(edge_vars.iter()) {
        objective.add_mul(edge.weight, *edge_var);
    }

    let mut model = variables.maximise(objective.clone()).using(highs);

    for edge_ids in color_incoming_edges(graph).values() {
        let mut color_expr = Expression::with_capacity(edge_ids.len());
        for edge_id in edge_ids {
            color_expr.add_mul(1.0, edge_vars[*edge_id]);
        }
        model.add_constraint(color_expr.leq(1.0));
    }

    for fragment_id in 1..graph.fragments.len() {
        if incoming[fragment_id].is_empty() || outgoing[fragment_id].is_empty() {
            continue;
        }
        for outgoing_edge_id in &outgoing[fragment_id] {
            let mut connected_expr = Expression::with_capacity(incoming[fragment_id].len() + 1);
            connected_expr.add_mul(1.0, edge_vars[*outgoing_edge_id]);
            for incoming_edge_id in &incoming[fragment_id] {
                connected_expr.add_mul(-1.0, edge_vars[*incoming_edge_id]);
            }
            model.add_constraint(connected_expr.leq(0.0));
        }
    }

    let mut root_expr = Expression::with_capacity(root_edges.len());
    for edge_id in root_edges {
        root_expr.add_mul(1.0, edge_vars[*edge_id]);
    }
    model.add_constraint(root_expr.eq(1.0));

    if let Some(minimal_score) = options.minimal_score {
        if minimal_score.is_finite() {
            model.add_constraint(objective.clone().geq(minimal_score));
        }
    }

    if let Some(seconds) = options.time_limit_seconds {
        if seconds.is_finite() && seconds >= 0.0 {
            model = model.with_time_limit(seconds);
        }
    }
    if let Some(threads) = options.threads {
        if threads > 0 {
            model = model.set_threads(threads);
        }
    }

    let solution = model.solve().map_err(|err| {
        let text = err.to_string();
        if text.contains("Infeasible") {
            TreeSolveError::Infeasible
        } else {
            TreeSolveError::NoSolution(text)
        }
    })?;

    let selected_edges: Vec<usize> = edge_vars
        .iter()
        .enumerate()
        .filter_map(|(edge_id, edge_var)| {
            if solution.value(*edge_var) > 0.5 {
                Some(edge_id)
            } else {
                None
            }
        })
        .collect();

    build_tree_from_selected_edges_from_good_lp_status(
        graph,
        &selected_edges,
        objective.eval_with(&solution),
        solution.status(),
    )
}

#[cfg(feature = "gurobi")]
fn solve_optimal_colorful_tree_gurobi(
    graph: &FragmentationGraph,
    options: TreeSolveOptions,
) -> Result<FragmentationTree, TreeSolveError> {
    use grb::expr::LinExpr;
    use grb::prelude::*;

    let incoming = graph.incoming_edges();
    let outgoing = graph.outgoing_edges();
    let root_edges = &outgoing[PSEUDO_ROOT_ID];
    if root_edges.is_empty() {
        return Err(TreeSolveError::InvalidGraph(
            "fragmentation graph has no pseudo-root outgoing edge".to_string(),
        ));
    }

    let mut model = create_gurobi_model("fragmentation_tree")?;
    model
        .set_param(param::OutputFlag, 0)
        .map_err(gurobi_error)?;
    model
        .set_param(param::LogToConsole, 0)
        .map_err(gurobi_error)?;
    if let Some(seconds) = options.time_limit_seconds {
        if seconds.is_finite() && seconds >= 0.0 {
            model
                .set_param(param::TimeLimit, seconds)
                .map_err(gurobi_error)?;
        }
    }
    if let Some(threads) = options.threads {
        if threads > 0 {
            model
                .set_param(param::Threads, threads as i32)
                .map_err(gurobi_error)?;
        }
    }

    let mut edge_vars = Vec::with_capacity(graph.edges.len());
    for edge_id in 0..graph.edges.len() {
        let name = format!("e{edge_id}");
        edge_vars.push(add_binvar!(model, name: &name).map_err(gurobi_error)?);
    }
    model.update().map_err(gurobi_error)?;

    let mut objective = LinExpr::new();
    for (edge, edge_var) in graph.edges.iter().zip(edge_vars.iter()) {
        objective.add_term(edge.weight, *edge_var);
    }
    model
        .set_objective(objective.clone(), Maximize)
        .map_err(gurobi_error)?;

    for (color, edge_ids) in color_incoming_edges(graph) {
        let mut color_expr = LinExpr::new();
        for edge_id in edge_ids {
            color_expr.add_term(1.0, edge_vars[edge_id]);
        }
        model
            .add_constr(
                &format!("color_{color}"),
                grb_constraint(color_expr, ConstrSense::Less, 1.0),
            )
            .map_err(gurobi_error)?;
    }

    for fragment_id in 1..graph.fragments.len() {
        if incoming[fragment_id].is_empty() || outgoing[fragment_id].is_empty() {
            continue;
        }
        for outgoing_edge_id in &outgoing[fragment_id] {
            let mut connected_expr = LinExpr::new();
            connected_expr.add_term(1.0, edge_vars[*outgoing_edge_id]);
            for incoming_edge_id in &incoming[fragment_id] {
                connected_expr.add_term(-1.0, edge_vars[*incoming_edge_id]);
            }
            model
                .add_constr(
                    &format!("connected_{outgoing_edge_id}"),
                    grb_constraint(connected_expr, ConstrSense::Less, 0.0),
                )
                .map_err(gurobi_error)?;
        }
    }

    let mut root_expr = LinExpr::new();
    for edge_id in root_edges {
        root_expr.add_term(1.0, edge_vars[*edge_id]);
    }
    model
        .add_constr(
            "one_root",
            grb_constraint(root_expr, ConstrSense::Equal, 1.0),
        )
        .map_err(gurobi_error)?;

    if let Some(minimal_score) = options.minimal_score {
        if minimal_score.is_finite() {
            model
                .add_constr(
                    "minimal_score",
                    grb_constraint(objective.clone(), ConstrSense::Greater, minimal_score),
                )
                .map_err(gurobi_error)?;
        }
    }

    model.optimize().map_err(gurobi_error)?;
    let status = model.status().map_err(gurobi_error)?;
    let solution_count: i32 = model.get_attr(attr::SolCount).map_err(gurobi_error)?;
    let tree_status = match status {
        Status::Optimal => TreeSolutionStatus::Optimal,
        Status::TimeLimit if solution_count > 0 => TreeSolutionStatus::TimeLimit,
        Status::Infeasible | Status::InfOrUnbd => return Err(TreeSolveError::Infeasible),
        Status::SubOptimal | Status::UserObjLimit if solution_count > 0 => {
            TreeSolutionStatus::GapLimit
        }
        _ if solution_count > 0 => TreeSolutionStatus::GapLimit,
        _ => {
            return Err(TreeSolveError::NoSolution(format!(
                "Gurobi returned status {status:?} with no incumbent solution"
            )))
        }
    };

    let values: Vec<f64> = model
        .get_obj_attr_batch(attr::X, edge_vars.iter().copied())
        .map_err(gurobi_error)?;
    let selected_edges: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(
            |(edge_id, value)| {
                if *value > 0.5 {
                    Some(edge_id)
                } else {
                    None
                }
            },
        )
        .collect();
    let solver_objective: f64 = model.get_attr(attr::ObjVal).map_err(gurobi_error)?;

    build_tree_from_selected_edges_with_status(
        graph,
        &selected_edges,
        solver_objective,
        tree_status,
    )
}

#[cfg(feature = "gurobi")]
fn create_gurobi_model(name: &str) -> Result<grb::Model, TreeSolveError> {
    use grb::prelude::*;

    std::thread_local! {
        static GUROBI_ENV: Result<Env, String> = {
            let mut env = Env::empty().map_err(|err| err.to_string())?;
            env.set(param::OutputFlag, 0).map_err(|err| err.to_string())?;
            env.set(param::LogToConsole, 0).map_err(|err| err.to_string())?;
            env.set(param::LogFile, "".to_string()).map_err(|err| err.to_string())?;
            env.start().map_err(|err| err.to_string())
        };
    }

    GUROBI_ENV.with(|env| match env {
        Ok(env) => Model::with_env(name, env).map_err(gurobi_error),
        Err(err) => Err(TreeSolveError::NoSolution(format!(
            "Gurobi environment error: {err}"
        ))),
    })
}

#[cfg(feature = "gurobi")]
fn grb_constraint(
    lhs: grb::expr::LinExpr,
    sense: grb::ConstrSense,
    rhs: f64,
) -> grb::constr::IneqExpr {
    grb::constr::IneqExpr {
        lhs: lhs.into(),
        sense,
        rhs: grb::Expr::Constant(rhs),
    }
}

#[cfg(feature = "gurobi")]
fn gurobi_error(err: grb::Error) -> TreeSolveError {
    let text = err.to_string();
    if text.contains("infeasible") || text.contains("Infeasible") {
        TreeSolveError::Infeasible
    } else {
        TreeSolveError::NoSolution(format!("Gurobi error: {text}"))
    }
}

#[cfg(not(feature = "gurobi"))]
fn solve_optimal_colorful_tree_gurobi(
    _graph: &FragmentationGraph,
    _options: TreeSolveOptions,
) -> Result<FragmentationTree, TreeSolveError> {
    Err(TreeSolveError::NoSolution(
        "Gurobi solver requested, but find-mfs-rust was built without the 'gurobi' feature"
            .to_string(),
    ))
}

fn build_tree_from_selected_edges_from_good_lp_status(
    graph: &FragmentationGraph,
    selected_edges: &[usize],
    solver_objective: f64,
    status: SolutionStatus,
) -> Result<FragmentationTree, TreeSolveError> {
    let status = match status {
        SolutionStatus::Optimal => TreeSolutionStatus::Optimal,
        SolutionStatus::TimeLimit => TreeSolutionStatus::TimeLimit,
        SolutionStatus::GapLimit => TreeSolutionStatus::GapLimit,
    };
    build_tree_from_selected_edges_with_status(graph, selected_edges, solver_objective, status)
}

fn build_tree_from_selected_edges_with_status(
    graph: &FragmentationGraph,
    selected_edges: &[usize],
    solver_objective: f64,
    status: TreeSolutionStatus,
) -> Result<FragmentationTree, TreeSolveError> {
    let root_edges: Vec<usize> = selected_edges
        .iter()
        .copied()
        .filter(|edge_id| graph.edges[*edge_id].source == PSEUDO_ROOT_ID)
        .collect();
    if root_edges.len() != 1 {
        return Err(TreeSolveError::NoSolution(format!(
            "expected exactly one selected pseudo-root edge, got {}",
            root_edges.len()
        )));
    }

    let root_fragment = graph.edges[root_edges[0]].target;
    let mut selected_outgoing = vec![Vec::new(); graph.fragments.len()];
    for edge_id in selected_edges {
        let edge = &graph.edges[*edge_id];
        selected_outgoing[edge.source].push(*edge_id);
    }

    let mut selected_fragments = Vec::new();
    let mut seen = vec![false; graph.fragments.len()];
    let mut queue = VecDeque::new();
    seen[root_fragment] = true;
    queue.push_back(root_fragment);

    while let Some(fragment_id) = queue.pop_front() {
        selected_fragments.push(fragment_id);
        for edge_id in &selected_outgoing[fragment_id] {
            let child = graph.edges[*edge_id].target;
            if !seen[child] {
                seen[child] = true;
                queue.push_back(child);
            }
        }
    }

    if selected_fragments.len() != selected_edges.len() {
        return Err(TreeSolveError::NoSolution(
            "selected edges do not form one connected tree".to_string(),
        ));
    }

    let tree_weight: f64 = selected_edges
        .iter()
        .map(|edge_id| graph.edges[*edge_id].weight)
        .sum();
    if (tree_weight - solver_objective).abs() > 1e-6 {
        return Err(TreeSolveError::NoSolution(format!(
            "reconstructed tree weight {tree_weight} differs from solver objective {solver_objective}"
        )));
    }

    Ok(FragmentationTree {
        root_fragment,
        selected_edges: selected_edges.to_vec(),
        selected_fragments,
        tree_weight,
        is_optimal: matches!(status, TreeSolutionStatus::Optimal),
        status,
    })
}

fn color_incoming_edges(graph: &FragmentationGraph) -> BTreeMap<i32, Vec<usize>> {
    let mut by_color = BTreeMap::new();
    for (edge_id, edge) in graph.edges.iter().enumerate() {
        let color = graph.fragments[edge.target].color;
        if color >= 0 {
            by_color.entry(color).or_insert_with(Vec::new).push(edge_id);
        }
    }
    by_color
}

fn optimistic_upper_bounds(graph: &FragmentationGraph) -> Result<Vec<f64>, TreeSolveError> {
    let outgoing = graph.outgoing_edges();
    let order = topological_order(graph)?;
    let mut upper = vec![0.0; graph.fragments.len()];

    for fragment_id in order.into_iter().rev() {
        let mut best_by_color: BTreeMap<i32, f64> = BTreeMap::new();
        for edge_id in &outgoing[fragment_id] {
            let edge = &graph.edges[*edge_id];
            let color = graph.fragments[edge.target].color;
            if color < 0 {
                continue;
            }
            let score = edge.weight + upper[edge.target];
            if score > 0.0 {
                best_by_color
                    .entry(color)
                    .and_modify(|best| *best = best.max(score))
                    .or_insert(score);
            }
        }
        upper[fragment_id] = best_by_color.values().sum();
    }

    Ok(upper)
}

fn topological_order(graph: &FragmentationGraph) -> Result<Vec<usize>, TreeSolveError> {
    let mut indegree = vec![0usize; graph.fragments.len()];
    let outgoing = graph.outgoing_edges();
    for edge in &graph.edges {
        indegree[edge.target] += 1;
    }

    let mut queue = VecDeque::new();
    for (fragment_id, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(fragment_id);
        }
    }

    let mut order = Vec::with_capacity(graph.fragments.len());
    while let Some(fragment_id) = queue.pop_front() {
        order.push(fragment_id);
        for edge_id in &outgoing[fragment_id] {
            let target = graph.edges[*edge_id].target;
            indegree[target] -= 1;
            if indegree[target] == 0 {
                queue.push_back(target);
            }
        }
    }

    if order.len() != graph.fragments.len() {
        return Err(TreeSolveError::InvalidGraph(
            "fragmentation graph must be acyclic".to_string(),
        ));
    }
    Ok(order)
}

pub fn is_subformula(parent: &[i32], child: &[i32]) -> bool {
    parent.len() == child.len()
        && parent
            .iter()
            .zip(child.iter())
            .all(|(parent_count, child_count)| *parent_count >= *child_count)
}

pub fn subtract_counts(parent: &[i32], child: &[i32]) -> Option<Vec<i32>> {
    if !is_subformula(parent, child) {
        return None;
    }
    Some(
        parent
            .iter()
            .zip(child.iter())
            .map(|(parent_count, child_count)| parent_count - child_count)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(formula: &str, counts: &[i32], color: i32, mass: f64) -> FragmentVertex {
        FragmentVertex {
            formula: formula.to_string(),
            counts: counts.to_vec(),
            ionization: "[M+H]+".to_string(),
            peak_id: Some(color as usize),
            color,
            mass,
        }
    }

    fn base_graph() -> FragmentationGraph {
        let mut graph = FragmentationGraph::new(2);
        graph.add_fragment(vertex("C4H8", &[4, 8], 0, 56.0));
        graph.add_fragment(vertex("C3H6", &[3, 6], 1, 42.0));
        graph.add_fragment(vertex("C2H4", &[2, 4], 1, 28.0));
        graph.add_fragment(vertex("CH2", &[1, 2], 2, 14.0));
        graph
    }

    fn candidate(
        formula: &str,
        counts: &[i32],
        peak_id: usize,
        color: i32,
        mass: f64,
        score: f64,
    ) -> FragmentCandidate {
        FragmentCandidate {
            formula: formula.to_string(),
            counts: counts.to_vec(),
            ionization: "[M+H]+".to_string(),
            peak_id,
            color,
            mass,
            score,
        }
    }

    #[test]
    fn subformula_helpers_check_and_subtract_counts() {
        assert!(is_subformula(&[4, 8, 2], &[3, 6, 0]));
        assert!(!is_subformula(&[4, 8], &[4, 9]));
        assert_eq!(subtract_counts(&[4, 8], &[3, 6]), Some(vec![1, 2]));
        assert_eq!(subtract_counts(&[4, 8], &[5, 1]), None);
    }

    #[test]
    fn subformula_graph_builder_adds_lazy_vertices_and_valid_losses() {
        let graph = build_subformula_graph(SubFormulaGraphInput {
            root_candidates: vec![candidate("C4H8", &[4, 8], 0, 0, 56.0, 1.0)],
            fragment_candidates: vec![
                candidate("C3H6", &[3, 6], 1, 1, 42.0, 3.0),
                candidate("C2H4", &[2, 4], 2, 2, 28.0, 5.0),
                candidate("C5H10", &[5, 10], 3, 3, 70.0, 100.0),
            ],
            allowed_ionizations: Vec::new(),
        })
        .unwrap();

        assert_eq!(
            graph
                .fragments
                .iter()
                .map(|fragment| fragment.formula.as_str())
                .collect::<Vec<_>>(),
            vec!["", "C4H8", "C3H6", "C2H4"]
        );
        assert_eq!(
            graph
                .edges
                .iter()
                .map(|edge| (edge.source, edge.target, edge.weight))
                .collect::<Vec<_>>(),
            vec![(0, 1, 1.0), (1, 2, 3.0), (1, 3, 5.0), (2, 3, 5.0)]
        );
    }

    #[test]
    fn subformula_graph_builder_applies_ionization_filter_and_loss_validator() {
        let mut sodium_child = candidate("C2H4", &[2, 4], 2, 2, 28.0, 5.0);
        sodium_child.ionization = "[M+Na]+".to_string();

        let graph = build_subformula_graph_with(
            SubFormulaGraphInput {
                root_candidates: vec![candidate("C4H8", &[4, 8], 0, 0, 56.0, 1.0)],
                fragment_candidates: vec![
                    candidate("C3H6", &[3, 6], 1, 1, 42.0, 3.0),
                    sodium_child,
                ],
                allowed_ionizations: vec!["[M+H]+".to_string()],
            },
            |parent, child| {
                if parent.formula == "C4H8" && child.formula == "C3H6" {
                    Some(child.score + 2.0)
                } else {
                    None
                }
            },
        )
        .unwrap();

        assert_eq!(graph.fragments.len(), 3);
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.edges[1].weight, 5.0);
    }

    #[test]
    fn graph_scoring_composes_sirius_style_edge_terms() {
        let mut scoring = GraphScoring::default();
        scoring.peak_scores.insert(0, 10.0);
        scoring.peak_scores.insert(1, 20.0);
        scoring.fragment_scores.insert("C4H8".to_string(), 30.0);
        scoring.fragment_scores.insert("C3H6".to_string(), 40.0);
        scoring.peak_pair_scores.insert((0, 1), 50.0);
        scoring
            .loss_scores
            .insert(("C4H8".to_string(), "C3H6".to_string()), 60.0);
        scoring.general_graph_score = 70.0;

        let graph = build_scored_subformula_graph(
            SubFormulaGraphInput {
                root_candidates: vec![candidate("C4H8", &[4, 8], 0, 0, 56.0, 1.0)],
                fragment_candidates: vec![candidate("C3H6", &[3, 6], 1, 1, 42.0, 2.0)],
                allowed_ionizations: Vec::new(),
            },
            &scoring,
        )
        .unwrap();

        assert_eq!(graph.edges[0].weight, 111.0);
        assert_eq!(graph.edges[1].weight, 172.0);
    }

    #[test]
    fn compute_fragmentation_tree_builds_scores_reduces_and_solves() {
        let mut scoring = GraphScoring::default();
        scoring.peak_scores.insert(1, 4.0);
        scoring.peak_scores.insert(2, 7.0);
        scoring.peak_pair_scores.insert((0, 1), 1.0);
        scoring.peak_pair_scores.insert((1, 2), 2.0);

        let result = compute_fragmentation_tree(
            SubFormulaGraphInput {
                root_candidates: vec![candidate("C4H8", &[4, 8], 0, 0, 56.0, 1.0)],
                fragment_candidates: vec![
                    candidate("C3H6", &[3, 6], 1, 1, 42.0, 3.0),
                    candidate("C2H4", &[2, 4], 2, 2, 28.0, 5.0),
                    candidate("CH2", &[1, 2], 3, 3, 14.0, -20.0),
                ],
                allowed_ionizations: Vec::new(),
            },
            &scoring,
            TreeSolveOptions::default(),
            true,
        )
        .unwrap();

        assert!(result.reduced_graph.is_some());
        assert!(result.tree.tree_weight > 20.0);
        let tree_graph = result.reduced_graph.as_ref().unwrap();
        let selected_formulas: Vec<_> = result
            .tree
            .selected_fragments
            .iter()
            .map(|fragment_id| tree_graph.fragments[*fragment_id].formula.as_str())
            .collect();
        assert_eq!(selected_formulas, vec!["C4H8", "C3H6", "C2H4"]);
    }

    #[test]
    fn ilp_selects_best_connected_colorful_subtree() {
        let mut graph = base_graph();
        graph.add_loss(0, 1, 1.0, false);
        graph.add_loss(1, 2, 4.0, false);
        graph.add_loss(1, 3, 6.0, false);
        graph.add_loss(2, 4, 10.0, false);
        graph.add_loss(3, 4, 1.0, false);

        let tree = solve_optimal_colorful_tree(&graph, TreeSolveOptions::default()).unwrap();

        assert_eq!(tree.root_fragment, 1);
        assert_eq!(tree.selected_edges, vec![0, 1, 3]);
        assert_eq!(tree.selected_fragments, vec![1, 2, 4]);
        assert_eq!(tree.tree_weight, 15.0);
        assert!(tree.is_optimal);
    }

    #[test]
    fn ilp_enforces_one_fragment_per_peak_color() {
        let mut graph = base_graph();
        graph.add_loss(0, 1, 0.0, false);
        graph.add_loss(1, 2, 5.0, false);
        graph.add_loss(1, 3, 4.0, false);

        let tree = solve_optimal_colorful_tree(&graph, TreeSolveOptions::default()).unwrap();

        assert_eq!(tree.selected_edges, vec![0, 1]);
        assert_eq!(tree.tree_weight, 5.0);
    }

    #[test]
    fn ilp_selects_exactly_one_root_candidate() {
        let mut graph = FragmentationGraph::new(2);
        graph.add_fragment(vertex("C4H8", &[4, 8], 0, 56.0));
        graph.add_fragment(vertex("C5H10", &[5, 10], 1, 70.0));
        graph.add_loss(0, 1, 3.0, false);
        graph.add_loss(0, 2, 5.0, false);

        let tree = solve_optimal_colorful_tree(&graph, TreeSolveOptions::default()).unwrap();

        assert_eq!(tree.root_fragment, 2);
        assert_eq!(tree.selected_edges, vec![1]);
        assert_eq!(tree.tree_weight, 5.0);
    }

    #[test]
    fn ilp_allows_negative_bridge_only_when_continuation_compensates() {
        let mut graph = base_graph();
        graph.add_loss(0, 1, 0.0, false);
        graph.add_loss(1, 2, -10.0, false);
        graph.add_loss(2, 4, 25.0, false);

        let tree = solve_optimal_colorful_tree(&graph, TreeSolveOptions::default()).unwrap();
        assert_eq!(tree.selected_edges, vec![0, 1, 2]);
        assert_eq!(tree.tree_weight, 15.0);

        graph.edges[1].weight = -30.0;
        let tree = solve_optimal_colorful_tree(&graph, TreeSolveOptions::default()).unwrap();
        assert_eq!(tree.selected_edges, vec![0]);
        assert_eq!(tree.tree_weight, 0.0);
    }

    #[test]
    fn minimal_score_can_make_tree_infeasible() {
        let mut graph = base_graph();
        graph.add_loss(0, 1, 0.0, false);

        let err = solve_optimal_colorful_tree(
            &graph,
            TreeSolveOptions {
                minimal_score: Some(1.0),
                ..TreeSolveOptions::default()
            },
        )
        .unwrap_err();

        assert!(matches!(
            err,
            TreeSolveError::Infeasible | TreeSolveError::NoSolution(_)
        ));
    }

    #[test]
    fn simple_reduction_removes_negative_dead_edges() {
        let mut graph = base_graph();
        graph.add_loss(0, 1, 0.0, false);
        graph.add_loss(1, 2, -1.0, false);
        graph.add_loss(1, 3, 2.0, false);
        graph.add_loss(2, 4, -0.5, false);

        let reduced = graph.simple_reduction().unwrap();

        assert_eq!(reduced.fragments.len(), 3);
        assert_eq!(reduced.edges.len(), 2);
        assert!(reduced
            .fragments
            .iter()
            .any(|fragment| fragment.formula == "C2H4"));
        assert!(!reduced
            .fragments
            .iter()
            .any(|fragment| fragment.formula == "C3H6"));
    }
}
