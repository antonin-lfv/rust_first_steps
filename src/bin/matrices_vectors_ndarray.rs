use ndarray::{array, Array1, Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform};

fn main() {
    // Créer un vecteur 1D (Array1)
    // Utilisation de la macro `array!` pour créer un tableau 1D
    let v: Array1<f64> = array![1.0, 2.0, 3.0];
    println!("Vecteur v = {}", v);

    // Créer une matrice 2D (Array2)
    let m: Array2<f64> = array![[1.0, 0.0, 2.0],
                                [0.0, 3.0, -1.0]];
    println!("Matrice m =\n{}", m);

    // Accès à un élément
    // En dimension n, on utilise de même m[[i1, i2, ..., in]] pour accéder à l'élément
    println!("m[1][2] = {}", m[[1, 2]]); // ligne 1, colonne 2 → -1.0

    // Accès à une ligne
    let ligne0 = m.row(0);
    println!("Ligne 0 = {}", ligne0);

    // Accès à une colonne
    let col1 = m.column(1);
    println!("Colonne 1 = {}", col1);

    // Modification d’un élément
    let mut m2 = m.clone();
    m2[[0, 2]] = 42.0;
    println!("Matrice modifiée =\n{}", m2);

    // Slicing
    let sub = m.slice(s![0..2, 1..]); // lignes 0 à 1, colonnes 1 à fin
    println!("Sous-matrice (slicing) =\n{}", sub);

    // Somme et moyenne sur une ligne
    let somme: f64 = m.row(0).sum();
    let moyenne = somme / m.ncols() as f64;
    println!("Somme ligne 0 = {}, Moyenne = {}", somme, moyenne);

    // Produit matrice-vecteur : m (2x3) · v (3x1) = (2x1)
    let result = m.dot(&v);
    println!("Produit m · v = {}", result);

    // Générer un vecteur 1D de taille 5 avec valeurs entre 0 et 1
    let v: Array1<f64> = Array1::random(5, Uniform::new(0.0, 1.0));
    println!("Vecteur aléatoire :\n{}\n", v);

    // Générer une matrice 2D de taille 5x5 avec valeurs entre 0 et 10
    let m = create_random_uniform_array2(5, 5, 0.0, 10.0);
    println!("Matrice aléatoire :\n{}", m);

    // on peut également utiliser d'autres distributions comme Normal, Exp, etc

    // Produit de la matrice m par le vecteur v
    // m (5x5) · v (5x1) = (5x1)
    let result = m.dot(&v);
    println!("Produit m · v =\n{}", result);

    // transposition de la matrice (créé une vue, pas une copie)
    let m_transpose = m.t();
    println!("Matrice transposée :\n{}", m_transpose);

    // transposition de la matrice (créé une copie)
    let m_transpose_copy = m.t().to_owned();
    println!("Matrice transposée (copie) :\n{}", m_transpose_copy);
}


fn create_random_uniform_array2(dimension: usize, size: usize, low: f64, high: f64) -> Array2<f64> {
    // Crée un tableau 2D de taille (dimension, size) avec des valeurs uniformément distribuées entre low et high
    Array2::random((dimension, size), Uniform::new(low, high))
}