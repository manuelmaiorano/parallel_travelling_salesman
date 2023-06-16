use std::collections::VecDeque;
use std::fs::File;
use std::io::prelude::*;
use std::sync::{Mutex, Arc, Condvar};
use std::{thread, vec};

use ndarray::Array2;

pub fn parse_file(filename: &str) -> std::io::Result<Array2<f64>> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let N = contents.split("\n").next().unwrap().split_whitespace().count();

    let mut array = Array2::zeros((N, N));

    for (i, line) in contents.split("\n").enumerate() {
        for (j, value) in line.split_whitespace().enumerate() {
            let value = value.trim().parse::<f64>().unwrap();
            array[[i, j]] = value;
        }
    }

    Ok(array)

}

#[derive(Clone)]
pub struct Tour {
    cities: Vec<usize>,
    cost: f64
}

impl Tour {
    pub fn is_feasible(&self, city_index: &usize) -> bool{
        for city in &self.cities {
            if city == city_index {
                return  false;
            }
        }
        return true;
    }

    pub fn add_city(&mut self, city_index: &usize, distance_matrix: &Array2<f64>) {
        let last_city = self.cities.pop().unwrap();
        let cost = distance_matrix[[last_city, *city_index]] + distance_matrix[[*city_index, 0]] 
                    - distance_matrix[[last_city, 0]];
        self.cities.push(last_city);
        self.cities.push(*city_index);
        self.cost += cost;

    }

    pub fn remove_last_city(&mut self, distance_matrix: &Array2<f64>) {

        if self.cities.len() == 1 {
            self.cities.pop().unwrap();
            self.cost = 0.0;
            return;
        }
        let last_city = self.cities.pop().unwrap();
        let second_to_last = self.cities.pop().unwrap();
        let cost = distance_matrix[[second_to_last, last_city]] + distance_matrix[[last_city, 0]];
        self.cost -= cost;
        self.cost += distance_matrix[[second_to_last, 0]];
        self.cities.push(second_to_last);
    }

}

pub fn serial_tsp_bb(distance_matrix: Array2<f64>) -> std::io::Result<Tour> {
    let mut stack: Vec<Tour> = Vec::new();
    let N = distance_matrix.dim().0;
    stack.push(Tour { cities: vec![0], cost: 0.0 });
    let mut best_tour = Tour { cities: vec![0], cost: f64::INFINITY };
    while !stack.is_empty() {
        let mut curr_tour = stack.pop().unwrap();
        if curr_tour.cost > best_tour.cost {
            continue;
        }
        if curr_tour.cities.len() == N {
            if curr_tour.cost < best_tour.cost {
                best_tour = curr_tour;
                let cost = best_tour.cost;
                let stack_size = stack.len();
                println!("cost: {cost}, stack_size: {stack_size}");
            }
        } else {
            for i in 0..N {
                if curr_tour.is_feasible(&i) {
                    curr_tour.add_city(&i, &distance_matrix);
                    stack.push(curr_tour.clone());
                    curr_tour.remove_last_city(&distance_matrix);
                }

            }
        }

    }
    Ok(best_tour)
}

fn get_initial_stack(distance_matrix: &Array2<f64>, thread_count: usize) -> (VecDeque<Tour>, Tour) {
    let mut queue = VecDeque::new();
    queue.push_back(Tour {cities: vec![0], cost: 0.0});
    let N = distance_matrix.dim().0;
    let mut best_tour = Tour { cities: vec![0], cost: f64::INFINITY };
    while queue.len() < thread_count {
        let mut curr_tour = queue.pop_front().unwrap();
        if curr_tour.cost > best_tour.cost {
            continue;
        }
        if curr_tour.cities.len() == N {
            if curr_tour.cost < best_tour.cost {
                best_tour = curr_tour;
            }
        } else {
            for i in 0..N {
                if curr_tour.is_feasible(&i) {
                    curr_tour.add_city(&i, &distance_matrix);
                    queue.push_back(curr_tour.clone());
                    curr_tour.remove_last_city(&distance_matrix);
                }

            }
        }
    }

    (queue, best_tour)
}

struct TempStack {
    stack: Vec<Tour>,
    threads_waiting: usize,
    cond_var: Condvar,
    mutex: Mutex<bool>
}

impl TempStack {
    fn new() -> TempStack {
        TempStack { 
            stack: vec![], 
            threads_waiting: 0, 
            cond_var: Condvar::new(), 
            mutex: Mutex::new(false) }
    }
}

pub fn parallel_tsp_bb(distance_matrix: Array2<f64>, thread_count: usize) -> std::io::Result<Tour> {
    let (mut initial_stack, mut best_tour) = get_initial_stack(&distance_matrix, thread_count);
    let stack_len = initial_stack.len();
    let tours_per_thread = stack_len / thread_count;
    let best_tour = Arc::new(Mutex::new(best_tour));
    let distance_matrix = Arc::new(distance_matrix);
    let N = distance_matrix.dim().0;

    let mut handles = vec![];
    for i in 0..thread_count {
        let mut local_stack = Vec::new();
        let best_tour = Arc::clone(&best_tour);
        let distance_matrix = Arc::clone(&distance_matrix);
        for _ in 0..tours_per_thread {
            if let Some(tour) = initial_stack.pop_front() {
                local_stack.push(tour);
            }
        }
        let handle = thread::spawn(move || {
            while !local_stack.is_empty() {
                let mut curr_tour = local_stack.pop().unwrap();
                {
                    let best_tour = best_tour.lock().unwrap();
                    if curr_tour.cost > best_tour.cost {
                        continue;
                    }
                }
                if curr_tour.cities.len() == N {
                    {
                        let mut best_tour = best_tour.lock().unwrap();
                        if curr_tour.cost < best_tour.cost {
                            *best_tour = curr_tour;
                        }
                    }
                } else {
                    for i in 0..N {
                        if curr_tour.is_feasible(&i) {
                            curr_tour.add_city(&i, &distance_matrix);
                            local_stack.push(curr_tour.clone());
                            curr_tour.remove_last_city(&distance_matrix);
                        }

                    }
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let best_tour = best_tour.lock().unwrap().clone();
    Ok(best_tour)
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_parse() {
        let filename = "data/15_cities.txt";

        let array = parse_file(filename).unwrap();
        println!("{:?}", array);
    }

    #[test]
    fn test_bb() {
        let filename = "data/simple.txt";

        let array = parse_file(filename).unwrap();
        let best = serial_tsp_bb(array).unwrap();
        let cost = best.cost;
        let cities: Vec<usize> = best.cities.iter().map(|x| x+1).collect();
        println!("{cost}");
        println!("{:?}", cities);
    }

    #[test]
    fn test_bb_15() {
        let filename = "data/15_cities.txt";

        let array = parse_file(filename).unwrap();
        let best = serial_tsp_bb(array).unwrap();
        let cost = best.cost;
        let cities: Vec<usize> = best.cities.iter().map(|x| x+1).collect();
        println!("{cost}");
        println!("{:?}", cities);
    }

    #[test]
    fn test_bb_26() {
        let filename = "data/26_cities.txt";

        let array = parse_file(filename).unwrap();
        let best = serial_tsp_bb(array).unwrap();
        let cost = best.cost;
        let cities: Vec<usize> = best.cities.iter().map(|x| x+1).collect();
        println!("{cost}");
        println!("{:?}", cities);
    }

    #[test]
    fn test_parallel() {
        let filename = "data/15_cities.txt";

        let array = parse_file(filename).unwrap();
        let best = parallel_tsp_bb(array, 2).unwrap();
        let cost = best.cost;
        let cities: Vec<usize> = best.cities.iter().map(|x| x+1).collect();
        println!("{cost}");
        println!("{:?}", cities);

    }
}