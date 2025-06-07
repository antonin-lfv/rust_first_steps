use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;
use ndarray::{Array2, s};
use serde::Deserialize;

// Représente une ligne du fichier CSV avec des noms de colonnes explicites
#[derive(Debug, Deserialize)]
struct IrisRow {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Chemin vers le fichier CSV
    let path = Path::new("data/iris.csv");

    // Ouvrir le fichier CSV
    let file = File::open(path)?;

    // Création d’un reader CSV avec en-têtes activés
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    // Vecteurs temporaires pour construire l'Array2
    let mut features: Vec<f64> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut n_rows = 0;

    for result in reader.deserialize::<IrisRow>() {
        let row = result?;
        features.push(row.sepal_length);
        features.push(row.sepal_width);
        features.push(row.petal_length);
        features.push(row.petal_width);
        labels.push(row.species);
        n_rows += 1;
    }

    let n_features = 4;
    let data: Array2<f64> = Array2::from_shape_vec((n_rows, n_features), features)?;

    // Affichage de la forme de la matrice : (nombre d'exemples, 4)
    println!("Shape des données : {:?}", data.dim());

    // Affichage des 5 premières lignes
    println!("Premières lignes :\n{:?}", data.slice(s![..5, ..]));

    // Affichage des 5 premiers labels (espèces)
    println!("Labels : {:?}", &labels[..5]);

    Ok(())
}
