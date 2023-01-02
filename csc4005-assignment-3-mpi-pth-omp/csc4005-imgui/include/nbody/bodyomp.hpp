//
// Created by schrodinger on 11/2/21.
//
#pragma once

#include <random>
#include <utility>

#define MAX_THREADS 4

class BodyPool {
    // provides in this way so that
    // it is easier for you to send a the vector with MPI
public:
    std::vector<double> x;
    std::vector<double> dx;
    std::vector<double> y;
    std::vector<double> dy;
    std::vector<double> vx;
    std::vector<double> dvx;
    std::vector<double> vy;
    std::vector<double> dvy;
    std::vector<double> ax;
    std::vector<double> dax;
    std::vector<double> ay;
    std::vector<double> day;
    std::vector<double> m;
    std::vector<int> acc_size;
    std::vector<int> start_pos;
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

    public:
        double &get_x() {
            return pool.x[index];
        }

        double &get_y() {
            return pool.y[index];
        }

        double &get_vx() {
            return pool.vx[index];
        }

        double &get_vy() {
            return pool.vy[index];
        }

        double &get_ax() {
            return pool.ax[index];
        }

        double &get_ay() {
            return pool.ay[index];
        }

        double &get_dvx() {
            return pool.dvx[index];
        }

        double &get_dvy() {
            return pool.dvy[index];
        }

        double &get_dax() {
            return pool.dax[index];
        }

        double &get_day() {
            return pool.day[index];
        }

        double &get_dx() {
            return pool.dx[index];
        }

        double &get_dy() {
            return pool.dy[index];
        }

        double &get_m() {
            return pool.m[index];
        }

        double distance_square(Body &that) {
            auto delta_x = get_x() - that.get_x();
            auto delta_y = get_y() - that.get_y();
            return delta_x * delta_x + delta_y * delta_y;
        }

        double distance(Body &that) {
            return std::sqrt(distance_square(that));
        }

        double delta_x(Body &that) {
            return get_x() - that.get_x();
        }

        double delta_y(Body &that) {
            return get_y() - that.get_y();
        }

        bool collide(Body &that, double radius) {
            return distance_square(that) <= radius * radius;
        }

        // collision with wall
        void handle_wall_collision(double position_range, double radius) {
            bool flag = false;
            if (get_x() <= radius) {
                flag = true;
                get_x() = radius + radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            } else if (get_x() >= position_range - radius) {
                flag = true;
                get_x() = position_range - radius - radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            }

            if (get_y() <= radius) {
                flag = true;
                get_y() = radius + radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            } else if (get_y() >= position_range - radius) {
                flag = true;
                get_y() = position_range - radius - radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            }
            if (flag) {
                get_ax() = 0;
                get_ay() = 0;
            }
        }

        void update_for_tick(
                double elapse,
                double position_range,
                double radius) {
            get_vx() += get_ax() * elapse;
            get_vy() += get_ay() * elapse;
            handle_wall_collision(position_range, radius);
            get_x() += get_vx() * elapse;
            get_y() += get_vy() * elapse;
            handle_wall_collision(position_range, radius);
        }

    };

    BodyPool(size_t size, double position_range, double mass_range, int thread_num) :
            x(size), dx(size), y(size), dy(size), vx(size), dvx(size), vy(size), dvy(size), ax(size), dax(size), ay(size), day(size), m(size), acc_size(thread_num), start_pos(thread_num) {
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

    Body get_body(size_t index) {
        return {index, *this};
    }

    void clear_acceleration() {
        ax.assign(m.size(), 0.0);
        ay.assign(m.size(), 0.0);
        dx.assign(m.size(), 0.0);
        dy.assign(m.size(), 0.0);
        dvx.assign(m.size(), 0.0);
        dvy.assign(m.size(), 0.0);
        dax.assign(m.size(), 0.0);
        day.assign(m.size(), 0.0);
    }

    size_t size() {
        return m.size();
    }

    int thread_num() {
        return acc_size.size();
    }

    static void check_and_update(Body i, Body j, double radius, double gravity) {
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance_square = i.distance_square(j);
        auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius) {
            distance_square = radius * radius;
        }
        auto distance = i.distance(j);
        if (distance < radius) {
            distance = radius;
        }
        if (i.collide(j, radius)) {
            auto dot_prod = delta_x * (i.get_vx() - j.get_vx())
                            + delta_y * (i.get_vy() - j.get_vy());
            auto scalar = 2 / (i.get_m() + j.get_m()) * dot_prod / distance_square;
            i.get_dvx() -= scalar * delta_x * j.get_m();
            i.get_dvy() -= scalar * delta_y * j.get_m();
            // j.get_vx() += scalar * delta_x * i.get_m();
            // j.get_vy() += scalar * delta_y * i.get_m();
            // now relax the distance a bit: after the collision, there must be
            // at least (ratio * radius) between them
            i.get_dx() += delta_x / distance * ratio * radius / 2.0;
            i.get_dy() += delta_y / distance * ratio * radius / 2.0;
            // j.get_x() -= delta_x / distance * ratio * radius / 2.0;
            // j.get_y() -= delta_y / distance * ratio * radius / 2.0;
            

        } else {
            // update acceleration only when no collision
            auto scalar = gravity / distance_square / distance;
            i.get_dax() -= scalar * delta_x * j.get_m();
            i.get_day() -= scalar * delta_y * j.get_m();            
            // j.get_ax() += scalar * delta_x * i.get_m();
            // j.get_ay() += scalar * delta_y * i.get_m();
        }
    }

    void update_for_tick(double elapse,
                         double gravity,
                         double position_range,
                         double radius) {
        
        clear_acceleration();

        omp_set_num_threads(thread_num());
        
        #pragma omp parallel
        {
	        int id = omp_get_thread_num();
	        int numthreads = omp_get_num_threads();

            #pragma omp master
            {
                int acc_pos = 0;
                for (int k = 0; k < numthreads; k++) {
                    acc_size[k] = get_length(size(), numthreads, k);
                    start_pos[k] = acc_pos;
                    acc_pos = acc_pos + acc_size[k];
                }
            
            }

            #pragma omp barrier

            for (size_t i = static_cast<size_t>(start_pos[id]); i < static_cast<size_t>(start_pos[id]+acc_size[id]); ++i) {
                for (size_t j = 0; j < size(); ++j) {
                    if (i != j) {
                        check_and_update(get_body(i), get_body(j), radius, gravity);
                    }

                }
            }

            #pragma omp barrier

	        for (size_t i = static_cast<size_t>(start_pos[id]); i < static_cast<size_t>(start_pos[id]+acc_size[id]); ++i) {
                if (dx[i] != 0) {
                    x[i] += dx[i];
                }
                if (dy[i] != 0) {
                    dy[i] += dy[i];
                }

                vx[i] += dvx[i];
                vy[i] += dvy[i];
                ax[i] += dax[i];
                ay[i] += day[i];
                get_body(i).update_for_tick(elapse, position_range, radius);
            }

        }

    }

    void print(int idx) {

        for (size_t i = 0; i < size(); i++) {
            Body body = get_body(i);
            std::cout << "i = " << idx << "; body = " << i << "; x = " << body.get_x() << "; y = " << body.get_y() <<"; vx = " << body.get_vx() << "; vy = " << body.get_vy() << "; ax = " << body.get_ax() <<"; ay = " << body.get_ay() << "; m = " << body.get_m() << std::endl;
            // std::cout << "i = " << idx << "; body = " << i << "; x = " << round(body.get_x()) << "; y = " << round(body.get_y()) << std::endl;
        }   

    }

    int get_length(int num, int thd, int idx) {
        return ((num - idx) % thd > 0) + (num - idx) / thd;
    }

};
