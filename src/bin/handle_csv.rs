use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;
use serde::Deserialize;

use ndarray::{Array2, s};
use nalgebra::DMatrix;

use polars::prelude::*;

// Représente une ligne du fichier CSV avec les bons noms de colonnes
#[derive(Debug, Deserialize)]
struct IrisRow {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = Path::new("data/iris.csv");

    // Chargement avec ndarray
    let (data_nd, labels_nd) = load_csv_ndarray(path)?;
    println!("[NDARRAY] Shape : {:?}", data_nd.dim());
    println!("[NDARRAY] Premières lignes :\n{:?}", data_nd.slice(s![..5, ..]));
    println!("[NDARRAY] Labels : {:?}", &labels_nd[..5]);

    // Chargement avec nalgebra
    let (data_na, labels_na) = load_csv_nalgebra(path)?;
    println!("[NALGEBRA] Shape : {} lignes × {} colonnes", data_na.nrows(), data_na.ncols());
    println!("[NALGEBRA] Premières lignes :\n{}", data_na.rows(0, 5));
    println!("[NALGEBRA] Labels : {:?}", &labels_na[..5]);

    // Chargement avec Polars
    let df = load_csv_polars("data/iris.csv")?;
    println!("[POLARS] Shape : {:?}", df.shape());
    println!("[POLARS] Colonnes : {:?}", df.get_column_names());
    println!("[POLARS] Premières lignes :\n{}", df.head(Some(5)));

    Ok(())
}

fn load_csv_ndarray(path: &Path) -> Result<(Array2<f64>, Vec<String>), Box<dyn Error>> {
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

    let data = Array2::from_shape_vec((n_rows, 4), features)?;
    Ok((data, labels))
}

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

    let data = DMatrix::from_vec(n_rows, 4, features);
    Ok((data, labels))
}

/// Charge un fichier CSV en un DataFrame Polars
///
/// # Arguments
/// * `path` - Le chemin vers le fichier CSV
///
/// # Returns
/// * `DataFrame` contenant les données
pub fn load_csv_polars(path: &str) -> PolarsResult<DataFrame> {
    use std::path::PathBuf;
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from(path)))?
        .finish()
}
