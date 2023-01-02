#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/bodycu.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}


double *arr_x = (double *)malloc(sizeof(double) * bodies);
double *arr_y = (double *)malloc(sizeof(double) * bodies);
double *arr_vx = (double *)malloc(sizeof(double) * bodies);
double *arr_vy = (double *)malloc(sizeof(double) * bodies);
double *arr_ax = (double *)malloc(sizeof(double) * bodies);
double *arr_ay = (double *)malloc(sizeof(double) * bodies);
double *arr_m = (double *)malloc(sizeof(double) * bodies);

double *d_arr_x, *d_arr_y, *d_arr_vx, *d_arr_vy, *d_arr_ax, *d_arr_ay, *d_arr_m;
cudaMalloc((void **)&d_arr_x, sizeof(double) * bodies);
cudaMalloc((void **)&d_arr_y, sizeof(double) * bodies);
cudaMalloc((void **)&d_arr_vx, sizeof(double) * bodies);
cudaMalloc((void **)&d_arr_vy, sizeof(double) * bodies);
cudaMalloc((void **)&d_arr_ax, sizeof(double) * bodies);
cudaMalloc((void **)&d_arr_ay, sizeof(double) * bodies);
cudaMalloc((void **)&d_arr_m, sizeof(double) * bodies);

__host__ int get_length(int num, int thd, int idx) {
    return ((num - idx) % thd > 0) + (num - idx) / thd;
}

__global__ void mykernel(
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

    pool_update_for_tick(
    d_arr_x,  
    d_arr_y, 
    d_arr_vx, 
    d_arr_vy,
    d_arr_ax,
    d_arr_ay,
    d_arr_m,
    idx1, 
    idx2, 
    elapse,
    gravity,
    position_range,
    radius,
    acc_size,
    start_pos);
}


int main(int argc, char **argv) {
    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    // static int bodies = 200;
    static int bodies = 5;
    static float elapse = 0.01;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    static int num_block = 1;
    static int num_thd = 1;

    int i = 0;
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
            num_thd = bodies / num_block;
            
            size_t *acc_size = (size_t *)malloc(sizeof(size_t) * num_thd); // the row_size of each rank
            size_t *start_pos = (size_t *)malloc(sizeof(size_t) * num_thd); // the start position of each rank

            int part_size;
            int acc_pos = 0;
            for (int i = 0; i < num_thd; i++) {
                acc_size[i] = get_length(bodies, num_thd, i);
                start_pos[i] = acc_pos;
                acc_pos = acc_pos + acc_size[i];
            }

            for (int i = 0; i < bodies; i++) {
                pool.ax[i] = 0;
                pool.ay[i] = 0;
                arr_x[i] = pool.x[i];
                arr_y[i] = pool.y[i];
                arr_vx[i] = pool.vx[i];
                arr_vy[i] = pool.vy[i];
                arr_ax[i] = pool.ax[i];
                arr_ay[i] = pool.ay[i];
                arr_m[i] = pool.m[i];
            }

            cudaMemcpy(d_arr_x, arr_x, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_y, arr_y, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_vx, arr_vx, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_vy, arr_vy, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_ax, arr_ax, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_ay, arr_ay, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_arr_m, arr_m, size, cudaMemcpyHostToDevice);

            mykernel<<<num_block,num_thread>>>(d_arr_x, d_arr_y, d_arr_vx, d_arr_vy, d_arr_ax, d_arr_ay, d_arr_m,
                idx1, idx2, elapse, gravity, position_range, radius, acc_size, start_pos);

            cudaMemcpy(arr_x, d_arr_x, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(arr_y, d_arr_y, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(arr_vx, d_arr_vx, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(arr_vy, d_arr_vy, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(arr_ax, d_arr_ax, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(arr_ay, d_arr_ay, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(arr_m, d_arr_m, size, cudaMemcpyDeviceToHost);
        
            for (size_t i = 0; i < bodies; ++i) {
                auto x = p.x + static_cast<float>(arr_x[i]);
                auto y = p.y + static_cast<float>(arr_y[i]);
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }


            // Cleanup
            free(arr_x); free(arr_y); free(arr_vx); free(arr_vy); free(arr_ax); free(arr_ay); free(arr_m); 
            cudaFree(d_arr_x); cudaFree(d_arr_y); cudaFree(d_arr_vx); cudaFree(d_arr_vy);cudaFree(d_arr_ax);cudaFree(d_arr_ay);cudaFree(d_arr_m);

        }
        ImGui::End();

        pool.print(i);

        i++;

        if (i == 60) {
            exit(0);
        }

    });
}
