#include <chrono>
#include <iostream>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <pthread.h>
#include <thread>
#include <cstring>
#include <nbody/bodypthread.hpp>

pthread_barrier_t barrier;

struct Arguments {
    BodyPool &pool;
    float elapse;
    float gravity;
    float space;
    float radius;
    int part_bodies;
    int start_pos;
    pthread_barrier_t &barrier;
};

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}
static constexpr size_t SHOW_THRESHOLD = 500000000ULL;

void *wrapper_update_for_tick(void *arg_ptr) {
    
    auto arguments = static_cast<Arguments *>(arg_ptr);
    arguments->pool.update_for_tick(arguments->elapse, arguments->gravity, arguments->space, arguments->radius, arguments->part_bodies, arguments->start_pos, arguments->barrier);
    delete arguments;
    return nullptr;

}

int get_length(int num, int thd, int idx) {
    return ((num - idx) % thd > 0) + (num - idx) / thd;
}

int main(int argc, char **argv) {
    UNUSED(argc, argv);


    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 200;
    // static int bodies = 5;
    static float elapse = 0.01;
    // static float elapse = 0.1;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);

        
    int n_bodies = 0; 
    size_t duration = 0;
    int i = 0;


    if (argc < 3) {
        std::cerr << "wrong arguments" << std::endl;
        std::cerr << "usage: " << argv[0] << " <integer-number of thread> <integer-number of bodies>" << std::endl;
        exit(1);
    }

    int thread_num;

    sscanf(argv[1], "%d", &thread_num);    
    printf("thread_num = %d\n", thread_num);

    sscanf(argv[2], "%d", &bodies);    
    printf("bodies = %d\n", bodies);
    

    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    graphic::GraphicContext context{"Assignment 2"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 2", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
        ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
        ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
        ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
        ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
        ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
        ImGui::ColorEdit4("Color", &color.x);

        if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
            space = current_space;
            bodies = current_bodies;
            max_mass = current_max_mass;
            pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
        }
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            
            auto begin = std::chrono::high_resolution_clock::now();
            int start_pos = 0;

            if (thread_num > bodies) {
                thread_num = bodies;
            }
            std::vector<pthread_t> threads(thread_num);
            pthread_barrier_init(&barrier, nullptr, thread_num);

            pool.clear_acceleration();

            for (unsigned long i = 0; i < threads.size(); ++i) {
                auto part_bodies = get_length(bodies, thread_num, i);
                pthread_create(&threads[i], nullptr, wrapper_update_for_tick, new Arguments{
                    pool,
                    elapse,
                    gravity,
                    space,
                    radius,
                    part_bodies,
                    start_pos,
                    barrier
                });
                start_pos += part_bodies;
            }
            
            for (auto & i : threads) {
                pthread_join(i, nullptr);
            }

            auto end = std::chrono::high_resolution_clock::now();

            n_bodies += bodies;
            duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();

            if (duration > SHOW_THRESHOLD) {
                std::cout << n_bodies << " bodies in last " << duration << " nanoseconds\n";
                auto speed = static_cast<double>(n_bodies) / static_cast<double>(duration) * 1e9;
                std::cout << "speed: " << speed << " bodies per second" << std::endl;
                n_bodies = 0;
                duration = 0;
            }

            for (size_t i = 0; i < pool.size(); ++i) {
                auto body = pool.get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End();

        // pool.print(i);

        i++;

        // if (i == 60) {
        // if (i == 1) {
            // exit(0);
        // }

    });
    
    

}

