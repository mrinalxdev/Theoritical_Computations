fn rk4(
    f: impl Fn(f64, f64) -> f64,
    t0: f64,
    y0: f64,
    h: f64,
    steps: usize
) -> Vec<(f64, f64)> {
    let mut result = Vec::with_capacity(steps + 1);
    let mut t = t0;
    let mut y = y0;

    result.push((t, y));

    for _ in 0..steps {
        let k1 = f(t, y);
        let k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
        let k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
        let k4 = f(t + h, y + h * k3);
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t = t + h;

        result.push((t, y));
    }

    result
}


fn main(){
    let f = |t : f64, y : f64| t * y;

    let t0 = 0.0;
    let y0 = 1.0;
    let h = 0.1;  // step size
    let steps = 10;

    let solution = rk4(f, t0, y0, h, steps);

    println!("t\t\ty");
    println!("-----------------");
    for (t, y) in solution {
        println!("{:.6}\t{:.6}", t, y);
    }
}
