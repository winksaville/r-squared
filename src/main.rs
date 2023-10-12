use nalgebra::DVector;

// Implement r_squared fuction for nalgebra::DVector
fn r_squared(y: &DVector<f64>, y_hat: &DVector<f64>) -> f64 {
    let y_mean = y.mean();
    let ss_tot = y
        .iter()
        .map(|y_i| {
            #[allow(clippy::let_and_return)]
            let a = (y_i - y_mean).powi(2);
            //println!("ss_tot: (y_i:{y_i} - y_mean:{y_mean})^2 = {a}");
            a
        })
        .sum::<f64>();
    let ss_res = y
        .iter()
        .zip(y_hat.iter())
        .map(|(y_i, y_hat_i)| {
            #[allow(clippy::let_and_return)]
            let r = (y_i - y_hat_i).powi(2);
            //println!("(y_i:{y_i} - y_hat_i:{y_hat_i})^2 = {r}");
            r
        })
        .sum::<f64>();

    #[allow(clippy::let_and_return)]
    let r2 = 1.0 - (ss_res / ss_tot);

    //println!("r_squared: {r2} = 1.0 - (ss_res:{ss_res} / ss_tot:{ss_tot})");
    r2
}

fn main() {
    // Example using r_squared
    let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_hat = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    println!("R^2: {}", r_squared(&y, &y_hat));
}

// Add test module
#[cfg(test)]
mod tests {
    use super::*;
    use approx;

    #[test]
    fn test_r_squared_1_0() {
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_hat = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(r_squared(&y, &y_hat), 1.0);
    }

    #[test]
    // From: https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html
    fn test_other() {
        let y = DVector::from_vec(vec![2.0, 4.0, 6.0, 7.0]);
        let y_hat = DVector::from_vec(vec![2.601, 3.83, 5.059, 7.517]);
        let rs = r_squared(&y, &y_hat);
        approx::assert_ulps_eq!(rs, 0.895, epsilon = 0.001);
    }
}
