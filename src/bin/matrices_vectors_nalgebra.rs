use nalgebra::{DVector, DMatrix};

fn main() {
    // Vecteur colonne de taille 3
    // DVector est un vecteur dynamique de taille variable
    let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);

    // Matrice 3x3
    // DMatrix est une matrice dynamique de taille variable, les 2 premiers paramètres sont les dimensions
    // et le dernier est un vecteur contenant les éléments de la matrice (colonne par colonne)
    let m = DMatrix::from_vec(3, 3, vec![
        1.0, 2.0, 0.0, // première colonne
        0.0, 1.0, 0.0, // deuxième colonne
        0.0, 0.0, 1.0, // troisième colonne
    ]);

    println!("Vecteur =\n{}", v);
    println!("Matrice =\n{}", m);

    // Produit matrice × vecteur
    let mv = &m * &v;
    println!("Produit matrice × vecteur =\n{}", mv);

    // Produit matrice × matrice (produit matriciel)
    let mm = &m * &m;
    println!("Produit matrice × matrice =\n{}", mm);

    // Création d'un vecteur avec repétition d'un élément
    // DVector::from_element crée un vecteur de taille donnée, rempli avec l'élément spécifié
    let v = DVector::from_element(4, 1.5); // vecteur [1.5, 1.5, 1.5, 1.5]

    // Création de la matrice identité de taille 4x4
    let identity_matrix = DMatrix::<f64>::identity(4, 4); // matrice identité 4x4
    println!("Matrice identité 4x4 =\n{}", identity_matrix);

    // Création d'une matrice nulle de taille 2x5
    let z = DMatrix::<f64>::zeros(2, 5); // matrice nulle
    println!("Matrice nulle 2x5 =\n{}", z);

    // Création d'une matrice constante de taille 2x2 remplie avec la valeur 42
    let r = DMatrix::<f64>::repeat(2, 2, 42.); // matrice constante
    println!("Matrice constante 2x2 avec 42 =\n{}", r);

    // Transposition de la matrice identité
    let transposed = identity_matrix.transpose();
    println!("Matrice transposée =\n{}", transposed);

    // Calcul de la norme d'un vecteur
    let norm = v.norm(); // norme L2 du vecteur
    println!("Norme du vecteur = {}", norm);

    // Calcul du produit scalaire d'un vecteur avec lui-même
    let dot = v.dot(&v); // produit scalaire
    println!("Produit scalaire du vecteur avec lui-même = {}", dot);
}
