use std::error::Error;
use std::fs::File;
use std::path::Path;
use serde::Deserialize;
use csv::ReaderBuilder;
use nalgebra::DMatrix;
use polars::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let path = Path::new("data/iris.csv");

    // 🔍 Analyse avec Polars
    let df = load_csv_polars(path.to_str().unwrap())?;
    println!("[POLARS] Premières lignes :\n{}", df.head(Some(5)));
    // TODO: Implémenter l'analyse avec Polars

    // 📈 Entraînement avec nalgebra
    let (data, labels): (DMatrix<f64>, Vec<String>) = load_csv_nalgebra(&path)?;
    println!("[NALGEBRA] Dimensions de la matrice : {}x{}", data.nrows(), data.ncols());
    // afficher les premières lignes
    print_head(&data, 5);
    // afficher les labels
    println!("[NALGEBRA] Labels : {:?}", &labels[..5]);
    // TODO: Implémenter l'entraînement avec nalgebra

    Ok(())
}

// Représente une ligne du fichier CSV avec les bons noms de colonnes
#[derive(Debug, Deserialize)]
struct IrisRow {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

/// Charge un fichier CSV en un DataFrame Polars
pub fn load_csv_polars(path: &str) -> PolarsResult<DataFrame> {
    use std::path::PathBuf;
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from(path)))?
        .finish()
}

/// Charge un fichier CSV en une matrice nalgebra et un vecteur de labels
fn load_csv_nalgebra(path: &Path) -> Result<(DMatrix<f64>, Vec<String>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut n_rows = 0;

    for result in reader.deserialize::<IrisRow>() {
        let row = result?;
        features.extend_from_slice(&[
            row.sepal_length,
            row.sepal_width,
            row.petal_length,
            row.petal_width,
        ]);
        labels.push(row.species);
        n_rows += 1;
    }

    let data = DMatrix::from_vec(4, n_rows, features).transpose();
    Ok((data, labels))
}

/// Affiche les `n` premières lignes d'une matrice
fn print_head(data: &DMatrix<f64>, n: usize) {
    println!("Premières lignes ({} premières) :", n);
    for i in 0..n.min(data.nrows()) {
        print!("{}", data.row(i));
    }
}
