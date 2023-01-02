#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <omp.h>
#include <nbody/bodyomp.hpp>

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}
static constexpr size_t SHOW_THRESHOLD = 500000000ULL;


int main(int argc, char **argv) {
    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 200;
    static float elapse = 0.01;
    int i = 0;
    int thread_num = 4;
    int n_bodies = 0; 
    size_t duration = 0;

    if (argc < 3) {
        std::cerr << "wrong arguments" << std::endl;
        std::cerr << "usage: " << argv[0] << " <integer-bodies> <integer-number of thread>" << std::endl;
        exit(1);
    }

    sscanf(argv[1], "%d", &bodies);    
    printf("bodies = %d\n", bodies);

    sscanf(argv[2], "%d", &thread_num);    
    printf("thread_num = %d\n", thread_num);


    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    graphic::GraphicContext context{"Assignment 2"};

    BodyPool pool(static_cast<size_t>(bodies), space, max_mass, thread_num);
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
            pool = BodyPool{static_cast<size_t>(bodies), space, max_mass, thread_num};
        }
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();

            auto begin = std::chrono::high_resolution_clock::now();
            pool.update_for_tick(elapse, gravity, space, radius);
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

        // if (i == 100) {
        // exit(0);
        // }

    });
}
