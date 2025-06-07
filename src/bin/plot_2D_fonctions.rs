use plotly::common::Mode;
use plotly::{Plot, Scatter};
use plotly::layout::{Layout, Axis};

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn main() {
    let x_vals: Vec<f64> = (-10..=10).map(|x| x as f64 * 0.5).collect();
    let y_vals_relu: Vec<f64> = x_vals.iter().map(|&x| relu(x)).collect();
    let y_vals_sigmoid: Vec<f64> = x_vals.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

    let trace = Scatter::new(x_vals.clone(), y_vals_relu)
        .mode(Mode::Lines)
        .name("ReLU");

    let mut plot = Plot::new();
    plot.add_trace(trace);

    let trace2 = Scatter::new(x_vals.clone(), y_vals_sigmoid).mode(Mode::Lines).name("Sigmoid");
    plot.add_trace(trace2);

    // Configuration de la mise en page du graphique
    plot.set_layout(
    Layout::new()
        .title("Fonctions d'activation")
        .x_axis(Axis::new().title("x"))
        .y_axis(Axis::new().title("f(x)")),
    );

    use std::process::Command;

    plot.write_html("plots/relu_plot.html");

    Command::new("open")
        .arg("relu_plot.html")
        .status()
        .expect("Échec de l'ouverture du fichier HTML");
}