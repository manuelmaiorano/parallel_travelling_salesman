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

pub fn serial_tsp_bb(distance_matrix: &Array2<f64>) -> std::io::Result<Tour> {
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
                    curr_tour.add_city(&i, distance_matrix);
                    stack.push(curr_tour.clone());
                    curr_tour.remove_last_city(distance_matrix);
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
    stack: Option<Vec<Tour>>,
    threads_waiting: usize
}

struct NewStack {
    cond_var: Condvar,
    mutex: Mutex<TempStack>
}

impl NewStack {
    fn new() -> NewStack {
        NewStack {
            cond_var: Condvar::new(),
            mutex: Mutex::new(TempStack { 
                stack: None,
                threads_waiting: 0
            })
        }
    }

    fn can_be_updated(&self) -> bool {
        let stack_data = self.mutex.lock().unwrap();
        return (stack_data.threads_waiting > 0) && (stack_data.stack.is_none());
    }
}

fn split_stack(stack: &mut Vec<Tour>) -> Vec<Tour> {
    let mut new_vec = vec![];
    for (i, tour) in stack.iter().enumerate() {
        if i % 2 != 0 {
            new_vec.push(tour.clone());
        }
    }
    let mut index = 0;
    stack.retain(|_| {
        let mut ret = false;
        if index % 2 == 0 {
            ret = true;
        }
        index += 1;
        ret
    });
    new_vec
}

fn is_terminated(stack: &mut Vec<Tour>, temp_stack: &Arc<NewStack>, thread_count: usize) -> bool {
    if (stack.len() > 2) && (temp_stack.can_be_updated()) {
        let mut stack_data = temp_stack.mutex.lock().unwrap();
        if (stack_data.threads_waiting > 0) && stack_data.stack.is_none(){
            stack_data.stack = Some(split_stack(stack));
            temp_stack.cond_var.notify_one();
        }
        return false;
    } else if  !stack.is_empty(){
        return false;
    } else {
        let mut stack_data = temp_stack.mutex.lock().unwrap();
        if stack_data.threads_waiting == thread_count - 1 {
            stack_data.threads_waiting += 1;
            temp_stack.cond_var.notify_all();
            return true;
        } else {
            stack_data.threads_waiting += 1;
            while stack_data.stack.is_none() && (stack_data.threads_waiting < thread_count) {
                stack_data = temp_stack.cond_var.wait(stack_data).unwrap();
            }
            if stack_data.threads_waiting < thread_count {
                for tour in stack_data.stack.take().unwrap() {
                    stack.push(tour);
                }
                stack_data.threads_waiting -= 1;
                return false;
            } else {
                return true;
            }

        }
    }
}

pub fn parallel_tsp_bb(distance_matrix: Array2<f64>, thread_count: usize) -> std::io::Result<Tour> {
    let (mut initial_stack, mut best_tour) = get_initial_stack(&distance_matrix, thread_count);
    let stack_len = initial_stack.len();
    let tours_per_thread = stack_len / thread_count;
    let best_tour = Arc::new(Mutex::new(best_tour));
    let distance_matrix = Arc::new(distance_matrix);
    let N = distance_matrix.dim().0;

    let mut new_stack = Arc::new(NewStack::new());
    let mut handles = vec![];
    for i in 0..thread_count {
        let mut local_stack = Vec::new();
        let best_tour = Arc::clone(&best_tour);
        let distance_matrix = Arc::clone(&distance_matrix);
        let new_stack = Arc::clone(&new_stack);
        for _ in 0..tours_per_thread {
            if let Some(tour) = initial_stack.pop_front() {
                local_stack.push(tour);
            }
        }
        let handle = thread::spawn(move || {
            while !is_terminated(&mut local_stack, &new_stack, thread_count) {
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
                            let cost = best_tour.cost;
                            let slen = local_stack.len();
                            println!("thread: {i} - cost: {cost} - stack_size: {slen}");
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
    use std::time::Instant;

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
        let best = serial_tsp_bb(&array).unwrap();
        let cost = best.cost;
        let cities: Vec<usize> = best.cities.iter().map(|x| x+1).collect();
        println!("{cost}");
        println!("{:?}", cities);
    }

    #[test]
    fn test_bb_15() {
        let filename = "data/15_cities.txt";

        let array = parse_file(filename).unwrap();
        let best = serial_tsp_bb(&array).unwrap();
        let cost = best.cost;
        let cities: Vec<usize> = best.cities.iter().map(|x| x+1).collect();
        println!("{cost}");
        println!("{:?}", cities);
    }

    #[test]
    fn test_bb_26() {
        let filename = "data/26_cities.txt";

        let array = parse_file(filename).unwrap();
        let best = serial_tsp_bb(&array).unwrap();
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

    #[test]
    fn test_compare() {
        let filename = "data/15_cities.txt";

        let array = parse_file(filename).unwrap();

        let now = Instant::now();
        serial_tsp_bb(&array);
        let duration = now.elapsed().as_secs();
        println!("serial - {duration}");

        let now = Instant::now();
        parallel_tsp_bb(array, 2);
        let duration = now.elapsed().as_secs();
        println!("parallel - {duration}");
    }
}