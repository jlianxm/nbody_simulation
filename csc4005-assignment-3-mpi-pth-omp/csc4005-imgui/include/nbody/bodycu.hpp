//
// Created by schrodinger on 11/2/21.
//
#pragma once

#include <random>
#include <utility>

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
    int idx1, 
    int idx2, 
    double elapse,
    double gravity,
    double position_range,
    double radius);
__host__ void print(int idx);

class BodyPool {
    // provides in this way so that
    // it is easier for you to send a the vector with MPI
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
    static constexpr double COLLISION_RATIO = 0.01;

    class Body {
        size_t index;
        BodyPool &pool;
        friend class BodyPool;
        Body(size_t index, BodyPool &pool) : index(index), pool(pool) {}
    };

public:

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
    auto delta_x = delta_x(idx1, idx2, d_arr_x);
    auto delta_y = delta_y(idx1, idx2, d_arr_y);
    return delta_x * delta_x + delta_y * delta_y;
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
        d_arr_x[idx] = radius + radius * COLLISION_RATIO;
        d_arr_vx[idx] = -d_arr_vx[idx];
    } else if (d_arr_x[idx] >= position_range - radius) {
        flag = true;
        d_arr_x[idx] = position_range - radius - radius * COLLISION_RATIO;
        d_arr_vx[idx] = -d_arr_vx[idx];
    }
    if (d_arr_y[idx] <= radius) {
        flag = true;
        d_arr_y[idx] = radius + radius * COLLISION_RATIO;
        d_arr_vy[idx] = -d_arr_vy[idx];
    } else if (d_arr_y[idx] >= position_range - radius) {
        flag = true;
        d_arr_y[idx] = position_range - radius - radius * COLLISION_RATIO;
        d_arr_vy[idx] = -d_arr_vy[idx];
    }
    if (flag) {
        d_arr_ay[idx] = 0;
        d_arr_ay[idx] = 0;
    }
}

__device__ void body_update_for_tick(
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
    d_arr_vx[idx] += d_arr_ax[idx] * elapse;
    d_arr_vy[idx] += d_arr_ay[idx] * elapse;
    handle_wall_collision(position_range, radius);
    d_arr_x[idx] += d_arr_vx[idx] * elapse;
    d_arr_y[idx] += d_arr_vy[idx] * elapse;
    handle_wall_collision(position_range, radius);
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
    
    auto delta_x = delta_x(idx1, idx2, d_arr_x);
    auto delta_y = delta_y(idx1, idx2, d_arr_y);
    auto distance_square = distance_square(idx1, idx2, d_arr_x, d_arr_y);
    auto ratio = 1 + COLLISION_RATIO;
    if (distance_square < radius * radius) {
        distance_square = radius * radius;
    }
    auto distance = distance(idx1, idx2, d_arr_x, d_arr_y);
    if (distance < radius) {
        distance = radius;
    }
    if (collide(idx1, idx2, d_arr_x, d_arr_y, radius)) {
        auto dot_prod = delta_x * (d_arr_vx[idx1] - d_arr_vx[idx2])
                        + delta_y * (d_arr_vy[idx1] - d_arr_vy[idx2]);
        auto scalar = 2 / (d_arr_m[idx1] + d_arr_m[idx2]) * dot_prod / distance_square;
        d_arr_vx[idx1] -= scalar * delta_x * d_arr_m[idx2];
        d_arr_vy[idx1] -= scalar * delta_y * d_arr_m[idx2];
        d_arr_vx[idx2] += scalar * delta_x * d_arr_m[idx1];
        d_arr_vy[idx2] += scalar * delta_y * d_arr_m[idx1];
        // now relax the distance a bit: after the collision, there must be
        // at least (ratio * radius) between them
        d_arr_x[idx1] += delta_x / distance * ratio * radius / 2.0;
        d_arr_y[idx1] += delta_y / distance * ratio * radius / 2.0;
        d_arr_x[idx2] -= delta_x / distance * ratio * radius / 2.0;
        d_arr_y[idx2] -= delta_y / distance * ratio * radius / 2.0;
    } else {
        // update acceleration only when no collision
        auto scalar = gravity / distance_square / distance;
        d_arr_ax[idx1] -= scalar * delta_x * d_arr_m[idx2];
        d_arr_ay[idx1] -= scalar * delta_y * d_arr_m[idx2];
        d_arr_ax[idx2] += scalar * delta_x * d_arr_m[idx1];
        d_arr_ay[idx2] += scalar * delta_y * d_arr_m[idx1];
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
    int idx1, 
    int idx2, 
    double elapse,
    double gravity,
    double position_range,
    double radius,
    size_t* acc_size,
    size_t* start_pos) {
    // ax.assign(size(), 0);
    // ay.assign(size(), 0);

    // get thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = start_pos[tid]; i < (start_pos[tid]+acc_size[tid]); ++i) {
        for (size_t j = i + 1; j < size(); ++j) {
            pool_check_and_update(d_arr_x, d_arr_y, d_arr_vx, d_arr_vy, d_arr_ax, d_arr_ay, d_arr_m, idx1, idx2, radius, gravity);
        }
    }

    for (size_t i = start_pos[tid]; i < (start_pos[tid]+acc_size[tid]); ++i) {
        body_update_for_tick(d_arr_x, d_arr_y, d_arr_vx, d_arr_vy, d_arr_ax, d_arr_ay, d_arr_m, elapse, position_range, radius);
    }
}

__host__ void print(int idx) {
    for (size_t i = 0; i < size(); i++) {
        // Body body = get_body(i);
        // std::cout << "i = " << idx << "; body = " << i << "; x = " << body.get_x() << "; y = " << body.get_y() <<"; vx = " << body.get_vx() << "; vy = " << body.get_vy() << "; ax = " << body.get_ax() <<"; ay = " << body.get_ay() << "; m = " << body.get_m() << std::endl;
        // std::cout << "i = " << idx << "; body = " << i << "; x = " << round(body.get_x()) << "; y = " << round(body.get_y()) << std::endl;
    }   
}

