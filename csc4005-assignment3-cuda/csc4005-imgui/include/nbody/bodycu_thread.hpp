//
// Created by schrodinger on 11/2/21.
//
#pragma once

#include <random>
#include <utility>
#include <stdio.h>

__device__ double delta_x(int idx1, int idx2, double* d_arr_x);
__device__ double delta_y(int idx1, int idx2, double* d_arr_y);
__device__ double distance_square(int idx1, int idx2, double* d_arr_x, double* d_arr_y);
__device__ double distance(int idx1, int idx2, double* d_arr_x, double* d_arr_y);
__device__ bool collide(int idx1, int idx2, double* d_arr_x, double* d_arr_y, double radius);
__device__ void handle_wall_collision(
        double* d_arr_x,  
        double* d_arr_y, 
        double* d_arr_vx, 
        double* d_arr_vy,
        double* d_arr_ax,
        double* d_arr_ay,
        double* d_arr_m,
        int idx, 
        double position_range, 
        double radius);
__device__ void body_update_for_tick(
        int idx,
        double* d_arr_x,  
        double* d_arr_y, 
        double* d_arr_vx, 
        double* d_arr_vy,
        double* d_arr_ax,
        double* d_arr_ay,
        double* d_arr_m,
        double elapse,
        double position_range,
        double radius);
__device__ void pool_check_and_update(
    double* d_arr_x,  
    double* d_arr_y, 
    double* d_arr_vx, 
    double* d_arr_vy,
    double* d_arr_ax,
    double* d_arr_ay,
    double* d_arr_m,
    int idx1, 
    int idx2, 
    double radius, 
    double gravity);
__device__ void pool_update_for_tick(
    double* d_arr_x,  
    double* d_arr_y, 
    double* d_arr_vx, 
    double* d_arr_vy,
    double* d_arr_ax,
    double* d_arr_ay,
    double* d_arr_m,
    int bodies,
    double elapse,
    double gravity,
    double position_range,
    double radius);
__host__ void print(int idx);

class BodyPool {
    // provides in this way so that
    // it is easier for you to send a the vector with MPI
public:
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> ax;
    std::vector<double> ay;
    std::vector<double> m;
    // so the movements of bodies are calculated discretely.
    // if after the collision, we do not separate the bodies a little bit, it may
    // results in strange outcomes like infinite acceleration.
    // hence, we will need to set up a ratio for separation.

    class Body {
        size_t index;
        BodyPool &pool;
        friend class BodyPool;
        Body(size_t index, BodyPool &pool) : index(index), pool(pool) {}
    };



    BodyPool(size_t size, double position_range, double mass_range) :
            x(size), y(size), vx(size), vy(size), ax(size), ay(size), m(size) {
        std::random_device device;
        std::default_random_engine engine{device()};
        engine.seed(1);
        std::uniform_real_distribution<double> position_dist{0, position_range};
        std::uniform_real_distribution<double> mass_dist{0, mass_range};
        for (auto &i : x) {
            i = position_dist(engine);
        }
        for (auto &i : y) {
            i = position_dist(engine);
        }
        for (auto &i : m) {
            i = mass_dist(engine);
        }
    }

};


__device__ double delta_x(int idx1, int idx2, double* d_arr_x) {
    return (d_arr_x[idx1] - d_arr_x[idx2]);
}

__device__ double delta_y(int idx1, int idx2, double* d_arr_y) {
    return (d_arr_y[idx1] - d_arr_y[idx2]);
}

__device__ double distance_square(int idx1, int idx2, double* d_arr_x, double* d_arr_y) {
    return (d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]);
}

__device__ double distance(int idx1, int idx2, double* d_arr_x, double* d_arr_y) {
    return std::sqrt(distance_square(idx1, idx2, d_arr_x, d_arr_y));
}

__device__ bool collide(int idx1, int idx2, double* d_arr_x, double* d_arr_y, double radius) {
    return distance_square(idx1, idx2, d_arr_x, d_arr_y) <= radius * radius;
}


__device__ void handle_wall_collision(
        double* d_arr_x,  
        double* d_arr_y, 
        double* d_arr_vx, 
        double* d_arr_vy,
        double* d_arr_ax,
        double* d_arr_ay,
        double* d_arr_m,
        int idx, 
        double position_range, 
        double radius) {
    bool flag = false;
    if (d_arr_x[idx] <= radius) {
        flag = true;
        d_arr_x[idx] = radius + radius * 0.01;
        d_arr_vx[idx] = -d_arr_vx[idx];
    } else if (d_arr_x[idx] >= position_range - radius) {
        flag = true;
        d_arr_x[idx] = position_range - radius - radius * 0.01;
        d_arr_vx[idx] = -d_arr_vx[idx];
    }
    if (d_arr_y[idx] <= radius) {
        flag = true;
        d_arr_y[idx] = radius + radius * 0.01;
        d_arr_vy[idx] = -d_arr_vy[idx];
    } else if (d_arr_y[idx] >= position_range - radius) {
        flag = true;
        d_arr_y[idx] = position_range - radius - radius * 0.01;
        d_arr_vy[idx] = -d_arr_vy[idx];
    }
    if (flag) {
        d_arr_ay[idx] = 0;
        d_arr_ay[idx] = 0;
    }
}

__device__ void body_update_for_tick(
        int idx,
        double* d_arr_x,  
        double* d_arr_y, 
        double* d_arr_vx, 
        double* d_arr_vy,
        double* d_arr_ax,
        double* d_arr_ay,
        double* d_arr_m,
        double elapse,
        double position_range,
        double radius) {
    d_arr_vx[idx] += d_arr_ax[idx] * (elapse);
    d_arr_vy[idx] += d_arr_ay[idx] * (elapse);
    handle_wall_collision(d_arr_x, d_arr_y, d_arr_vx, d_arr_vy, d_arr_ax, d_arr_ay,
                        d_arr_m, idx, position_range, radius);
    d_arr_x[idx] += d_arr_vx[idx] * (elapse);
    d_arr_y[idx] += d_arr_vy[idx] * (elapse);
    handle_wall_collision(d_arr_x, d_arr_y, d_arr_vx, d_arr_vy, d_arr_ax, d_arr_ay,
                        d_arr_m, idx, position_range, radius);
}


__device__ void pool_check_and_update(
    double* d_arr_x,  
    double* d_arr_y, 
    double* d_arr_vx, 
    double* d_arr_vy,
    double* d_arr_ax,
    double* d_arr_ay,
    double* d_arr_m,
    int idx1, 
    int idx2, 
    double radius, 
    double gravity) {

    // printf("enter check!!\n");
    
    // double distance_square = ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]));
    // double distance = std::sqrt(((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])));
    // double ratio = 1 + 0.01;
    if (((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])) < ((radius) * (radius))) {
        // ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])) = ((*radius) * (*radius));
    }

    if (std::sqrt(((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]))) < (radius)) {
        // std::sqrt(((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]))) = (*radius);
    }

    if (((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])) <= ((radius) * (radius))) {
        // double dot_prod = (d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_vx[idx1] - d_arr_vx[idx2])
        //                 + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_vy[idx1] - d_arr_vy[idx2]);
        // double scalar = 2 / ((d_arr_m[idx1] + d_arr_m[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_vx[idx1] - d_arr_vx[idx2])
        //                 + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_vy[idx1] - d_arr_vy[idx2])) / ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]));
        d_arr_vx[idx1] -= 2 / (d_arr_m[idx1] + d_arr_m[idx2]) * ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_vx[idx1] - d_arr_vx[idx2])
                        + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_vy[idx1] - d_arr_vy[idx2])) / (radius * radius) * (d_arr_x[idx1] - d_arr_x[idx2]) * d_arr_m[idx2];
        d_arr_vy[idx1] -= 2 / (d_arr_m[idx1] + d_arr_m[idx2]) * ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_vx[idx1] - d_arr_vx[idx2])
                        + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_vy[idx1] - d_arr_vy[idx2])) / (radius * radius) * (d_arr_y[idx1] - d_arr_y[idx2]) * d_arr_m[idx2];
        // d_arr_vx[idx2] += scalar * (d_arr_x[idx1] - d_arr_x[idx2]) * d_arr_m[idx1];
        // d_arr_vy[idx2] += scalar * (d_arr_y[idx1] - d_arr_y[idx2]) * d_arr_m[idx1];
        // now relax the distance a bit: after the collision, there must be
        // at least (ratio * radius) between them
        d_arr_x[idx1] += (d_arr_x[idx1] - d_arr_x[idx2]) / radius * 1.01 * (radius) / 2.0;
        d_arr_y[idx1] += (d_arr_y[idx1] - d_arr_y[idx2]) / radius * 1.01 * (radius) / 2.0;
        // d_arr_x[idx2] -= (d_arr_x[idx1] - d_arr_x[idx2]) / distance * ratio * radius / 2.0;
        // d_arr_y[idx2] -= (d_arr_y[idx1] - d_arr_y[idx2]) / distance * ratio * radius / 2.0;
    } else {
        // printf("enter no! check!!\n");
        // // update acceleration only when no collision
        // // double scalar = (*gravity) / ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])) / std::sqrt(((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])));
        // printf("gravity = %f\n", (gravity));
        // printf("distancq = %f\n", ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])));
        // printf("d = %f\n", std::sqrt(((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]))));
        // printf("d_arr_ax[idx1] = %f", d_arr_ax[idx1]);
        // printf("scaler = %f", (gravity) / ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])) / std::sqrt(((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]))));


        d_arr_ax[idx1] -= (gravity) / ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])) / std::sqrt(((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]))) * (d_arr_x[idx1] - d_arr_x[idx2]) * d_arr_m[idx2];
        d_arr_ay[idx1] -= (gravity) / ((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2])) / std::sqrt(((d_arr_x[idx1] - d_arr_x[idx2]) * (d_arr_x[idx1] - d_arr_x[idx2]) + (d_arr_y[idx1] - d_arr_y[idx2]) * (d_arr_y[idx1] - d_arr_y[idx2]))) * (d_arr_y[idx1] - d_arr_y[idx2]) * d_arr_m[idx2];
        // printf("d_arr_ax[idx1] = %f", d_arr_ax[idx1]);
        // d_arr_ax[idx2] += scalar * (d_arr_x[idx1] - d_arr_x[idx2]) * d_arr_m[idx1];
        // d_arr_ay[idx2] += scalar * (d_arr_y[idx1] - d_arr_y[idx2]) * d_arr_m[idx1];
    }
}

__device__ void pool_update_for_tick(
    double* d_arr_x,  
    double* d_arr_y, 
    double* d_arr_vx, 
    double* d_arr_vy,
    double* d_arr_ax,
    double* d_arr_ay,
    double* d_arr_m,
    int bodies,
    double elapse,
    double gravity,
    double position_range,
    double radius,
    int* acc_size,
    int* start_pos) {


    // get thread id
    int tid = threadIdx.x;
    // printf("enter pool!! tid = %d\n", tid);
    __syncthreads();

    for (int i = start_pos[tid]; i < (start_pos[tid]+acc_size[tid]); ++i) {
        for (int j = 0; j < bodies; ++j) {

            // printf("i = %d, start_pos[tid] = %d, j = %d, (start_pos[tid]+acc_size[tid]) = %d, num_bodies = %d, tid = %d\n", i, start_pos[tid],j, (start_pos[tid]+acc_size[tid]), bodies, tid);
            if (i != j) {
                // printf("n_gra = %f\n", gravity);
                pool_check_and_update(d_arr_x, d_arr_y, d_arr_vx, d_arr_vy, d_arr_ax, d_arr_ay, d_arr_m, i, j, radius, gravity);
            }
        }
    }

    __syncthreads();


    for (int i = start_pos[tid]; i < (start_pos[tid]+acc_size[tid]); ++i) {
        body_update_for_tick(i, d_arr_x, d_arr_y, d_arr_vx, d_arr_vy, d_arr_ax, d_arr_ay, d_arr_m, elapse, position_range, radius);
    }
    __syncthreads();
}

__host__ void print(int idx) {
    // for (size_t i = 0; i < size(); i++) {
        // Body body = get_body(i);
        // std::cout << "i = " << idx << "; body = " << i << "; x = " << body.get_x() << "; y = " << body.get_y() <<"; vx = " << body.get_vx() << "; vy = " << body.get_vy() << "; ax = " << body.get_ax() <<"; ay = " << body.get_ay() << "; m = " << body.get_m() << std::endl;
        // std::cout << "i = " << idx << "; body = " << i << "; x = " << round(body.get_x()) << "; y = " << round(body.get_y()) << std::endl;
    // }   
}

