use std::error::Error;
use std::fs::File;
use std::path::Path;
use serde::Deserialize;
use csv::ReaderBuilder;
use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use plotly::{Plot, Scatter};
use plotly::common::Mode;

fn main() -> Result<(), Box<dyn Error>> {
    let path = Path::new("data/housing.csv");

    // 🔍 Analyse avec Polars
    let df = load_csv_polars(path.to_str().unwrap())?;
    plot_numeric_pairplot_polars(&df, "plots")?;

    // 📈 Entraînement avec nalgebra
    let (data, target): (DMatrix<f64>, Vec<f64>) = load_csv_nalgebra(&path)?;
    let y_all = DVector::from_vec(target.clone());

    // --- Modèle multivarié (area + distance)
    let x_all = add_bias_column(&data);
    let theta_all = linear_regression_svd(&x_all, &y_all);
    println!("Paramètres du modèle (theta complet) :\n{}", theta_all);

    // --- Modèle univarié (juste "area")
    let x_feat = data.column(0).into_owned();             // DVector<f64>
    let x_uni_mat = DMatrix::from_row_slice(x_feat.len(), 1, x_feat.as_slice());
    let x_uni = add_bias_column(&x_uni_mat);
    println!("shape de x_uni : {:?}", x_uni.shape());
    println!("shape de x_feat : {:?}", x_feat.shape());

    let theta_uni = linear_regression_svd(&x_uni, &y_all);
    println!("Paramètres du modèle avec une seule feature (area) :\n{}", theta_uni);
    // ici on trouve [7594.85, 493.49] c'est à dire que la droite de régression est :
    // y = 7594.85 + 493.49 * area
    plot_regression_result(&x_feat, &y_all, &theta_uni, "area");

    Ok(())
}

/*
MODEL
*/

// Entraîne un modèle de régression linéaire en utilisant la SVD
fn linear_regression_svd(x: &DMatrix<f64>, y: &DVector<f64>) -> DVector<f64> {
    let svd = x.clone().svd(true, true);

    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let sigma = svd.singular_values;

    let sigma_pinv = DMatrix::from_diagonal(
        &sigma.map(|s| if s.abs() > 1e-10 { 1.0 / s } else { 0.0 })
    );

    let u_t_y = u.transpose() * y;
    let theta = v_t.transpose() * sigma_pinv * u_t_y;
    theta
}

// Construit la matrice de design X avec une colonne de biais
fn add_bias_column(x: &DMatrix<f64>) -> DMatrix<f64> {
    let (n_rows, n_cols) = x.shape();
    let mut data = Vec::with_capacity(n_rows * (n_cols + 1));
    for i in 0..n_rows {
        data.push(1.0);
        for j in 0..n_cols {
            data.push(x[(i, j)]);
        }
    }
    DMatrix::from_row_slice(n_rows, n_cols + 1, &data)
}

/*
DATA MANAGEMENT
*/

#[derive(Debug, Deserialize)]
struct HousingRow {
    price: f64,
    area: f64,
    distance_from_center: f64,
}

fn load_csv_polars(path: &str) -> PolarsResult<DataFrame> {
    use std::path::PathBuf;
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from(path)))?
        .finish()
}

fn load_csv_nalgebra(path: &Path) -> Result<(DMatrix<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut n_rows = 0;

    for result in reader.deserialize::<HousingRow>() {
        let row = result?;
        features.extend_from_slice(&[row.area, row.distance_from_center]);
        labels.push(row.price);
        n_rows += 1;
    }

    let data = DMatrix::from_row_slice(n_rows, 2, &features);
    Ok((data, labels))
}

fn print_head(data: &DMatrix<f64>, n: usize) {
    println!("Premières lignes ({} premières) :", n);
    for i in 0..n.min(data.nrows()) {
        print!("{}", data.row(i));
    }
}

/*
PLOTTING
*/

fn plot_numeric_pairplot_polars(df: &DataFrame, out_dir: &str) -> PolarsResult<()> {
    let numeric_cols: Vec<&Series> = df
        .iter()
        .filter(|s| matches!(s.dtype(), DataType::Float64))
        .collect();

    for (i, col_x) in numeric_cols.iter().enumerate() {
        for (j, col_y) in numeric_cols.iter().enumerate() {
            if i >= j {
                continue;
            }

            let x_vals: Vec<f64> = col_x.f64()?.into_no_null_iter().collect();
            let y_vals: Vec<f64> = col_y.f64()?.into_no_null_iter().collect();

            let trace = Scatter::new(x_vals, y_vals)
                .mode(Mode::Markers)
                .name(&format!("{} vs {}", col_x.name(), col_y.name()));

            let mut plot = Plot::new();
            plot.add_trace(trace);
            plot.set_layout(
                plotly::layout::Layout::new()
                    .title(&format!("{} vs {}", col_y.name(), col_x.name()))
                    .x_axis(plotly::layout::Axis::new().title(col_x.name().to_string()))
                    .y_axis(plotly::layout::Axis::new().title(col_y.name().to_string()))
            );

            let filename = format!("{}/{}_vs_{}.html", out_dir, col_x.name(), col_y.name());
            plot.write_html(filename);
        }
    }

    Ok(())
}

// Visualise les résultats de la régression
fn plot_regression_result(x: &DVector<f64>, y_true: &DVector<f64>, theta: &DVector<f64>, name: &str) {
    // Prédiction : y_pred = θ₀ + θ₁ * x
    let y_pred: Vec<f64> = x.iter().map(|xi| theta[0] + theta[1] * xi).collect();

    println!("x: {:?}", x);
    println!("y_true: {:?}", y_true);
    let trace_points = Scatter::new(x.as_slice().to_vec(), y_true.as_slice().to_vec())
        .mode(Mode::Markers)
        .name("Données");

    let trace_line = Scatter::new(x.as_slice().to_vec(), y_pred)
        .mode(Mode::Lines)
        .name("Régression");

    let mut plot = Plot::new();
    plot.add_trace(trace_points);
    plot.add_trace(trace_line);
    plot.set_layout(
        plotly::Layout::new()
            .title(format!("Régression sur {}", name))
            .x_axis(plotly::layout::Axis::new().title(name.to_string()))
            .y_axis(plotly::layout::Axis::new().title("Prix")),
    );
    plot.write_html(format!("plots/regression_{}.html", name));
}