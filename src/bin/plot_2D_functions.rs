use plotly::common::Mode;
use plotly::{Plot, Scatter};
use plotly::layout::{Layout, Axis};
use std::process::Command;

/// Fonction d'activation ReLU
/// ReLU (Rectified Linear Unit) est définie comme f(x) = max(0, x)
/// Elle est souvent utilisée dans les réseaux de neurones pour introduire de la non-linéarité
/// et permettre aux modèles d'apprendre des relations complexes dans les données.
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn main() {
    // On crée un vecteur de f64 contenant les valeurs de -5.0 à 5.0, espacées de 0.5
    // -10..=10 -> crée un itérateur de -10 à 10 inclus (-10, -9, -8, ..., 0, ..., 9, 10) ce sont des i32 !
    // la méthode collect() transforme cet itérateur en un vecteur Vec<i32>, cela consomme l'itérateur
    let x_vals: Vec<f64> = (-10..=10).map(|x| x as f64 * 0.5).collect();

    // On applique la fonction ReLU et Sigmoid à chaque valeur de x_vals
    // iter() crée un itérateur sur les références des éléments de x_vals, donc itération sur des &f64
    // x_vals est juste emprunté ici, pas consommé
    let y_vals_relu: Vec<f64> = x_vals.iter().map(|&x| relu(x)).collect();
    let y_vals_sigmoid: Vec<f64> = x_vals.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

    // On crée un nouveau graphique Plotly
    let mut plot = Plot::new();

    // On crée un graphique de type Scatter
    // On passe un clone de x_vals car Plotly n'accepte pas les références (les données doivent être possédées)
    // y_vals_relu n'est pas cloné car on ne l'utilise qu'une seule fois
    let trace1 = Scatter::new(x_vals.clone(), y_vals_relu)
        .mode(Mode::Lines)
        .name("ReLU");

    let trace2 = Scatter::new(x_vals.clone(), y_vals_sigmoid)
        .mode(Mode::Lines)
        .name("Sigmoid");

    // On ajoute les traces au graphique
    plot.add_trace(trace1);
    plot.add_trace(trace2);

    // Configuration de la mise en page du graphique
    plot.set_layout(
        Layout::new()
            .title("Fonctions d'activation")
            .x_axis(Axis::new().title("x"))
            .y_axis(Axis::new().title("f(x)")),
    );

    // On crée un fichier HTML contenant le graphique
    plot.write_html("plots/relu_plot.html");

    // On ouvre le fichier HTML dans le navigateur par défaut
    // Command permet d'exécuter une commande système, ici open
    // args permet de spécifier les arguments de la commande, ici le fichier HTML à ouvrir
    // status() exécute la commande et retourne un résultat
    // expect() permet de gérer l'erreur si la commande échoue
    Command::new("open")
        .arg("plots/relu_plot.html")
        .status()
        .expect("Échec de l'ouverture du fichier HTML");

}