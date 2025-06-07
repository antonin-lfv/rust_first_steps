use ndarray::Array1;

fn main() {
    println!("Here we'll calculate the Euclidean distance between two points in 2D space.");
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 6.0, 3.0];

    let distance = euclidean_distance(&a, &b);
    println!("Euclidean distance with slices : {}", distance);

    // Utilisation de ndarray
    let a_nd = Array1::from_vec(a.clone()); // on clone a pour éviter de consommer `a`
    let b_nd = Array1::from_vec(b.clone()); // on clone b pour éviter de consommer `b`
    let distance_nd = euclidean_distance_nd(&a_nd, &b_nd);
    println!("Euclidean distance with ndarray : {}", distance_nd);
}

/// Calcule la distance euclidienne entre deux vecteurs de même taille en utilisant les slices.
/// 
/// # Arguments
/// * `x` - Le premier vecteur, qui est une reférence à un slice de f64.
/// * `y` - Le second vecteur, qui est une référence à un slice de f64.
/// 
/// # Returns
/// * La distance euclidienne entre les deux vecteurs.
///
/// # Example
/// ```
/// let x = vec![1.0, 2.0];
/// let y = vec![4.0, 6.0];
/// let dist = euclidean_distance(&x, &y);
/// assert!((dist - 5.0).abs() < 1e-6);
/// ```
fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 { // &[f64] = slice = un vecteur dont on connaît la taille à l’exécution
    assert_eq!(x.len(), y.len(), "Les vecteurs doivent avoir la même taille !");
    let sum_squared_diff: f64 = x.iter() // .iter() = itère sur les éléments du vecteur
        .zip(y.iter()) // .zip() = associe les éléments des deux vecteurs
        .map(|(xi, yi)| (xi - yi).powi(2))  // powi(2) = carré d’un f64
        .sum();

    sum_squared_diff.sqrt()
}

/// Calcule la distance euclidienne entre deux vecteurs de même taille en utilisant ndarray.
fn euclidean_distance_nd(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    ((x - y).mapv(|v| v.powi(2))).sum().sqrt()  // mapv = applique une fonction à chaque élément du tableau (pas la copie de l'élément)
    // ou (x - y).mapv(|v| v.powi(2)).scalar_sum().sqrt()
}