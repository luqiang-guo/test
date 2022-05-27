#include <iostream>
#include <string>

#define NUM_UINT32 360
#define NUM_GROUP_DATA 50

struct Group_data
{
    float a;
    uint16_t b;
    uint16_t c;
};


int main(int argc, char** argv) {
    // std::string file_name = argv[1];

    FILE* file_handdle = fopen(argv[1], "rb");
    if (nullptr == file_handdle) {
        printf("file open failed!\n");
        return -1;
    }

    uint32_t* uint_data = (uint32_t*)malloc(NUM_UINT32 * sizeof(uint32_t));
    fread(uint_data, sizeof(uint32_t), NUM_UINT32, file_handdle);
    // fgets((char*)uint_data, NUM_UINT32 * sizeof(uint32_t), file_handdle);
    for (size_t i = 0; i < NUM_UINT32; i++) {
        std::cout << uint_data[i] << std::endl;
    }

    Group_data* group_data = (Group_data*)malloc(NUM_GROUP_DATA * sizeof(Group_data));
    fread(group_data, sizeof(Group_data), NUM_GROUP_DATA, file_handdle);
    // fgets((char*)group_data, NUM_GROUP_DATA * sizeof(Group_data), file_handdle);
    for (size_t i = 0; i < NUM_GROUP_DATA; i++) {
        printf("%.2f  ", group_data[i].a);
        std::cout << group_data[i].b << "  " << group_data[i].c << std::endl;
    }
    // for (size_t i = 0; i < NUM_GROUP_DATA; i++) {
    //     float data_float;
    //     uint16_t data_uint_1, data_uint_2;
    //     fgets((char*)&data_float, 4, file_handdle);
    //     fgets((char*)&data_uint_1, 2, file_handdle);
    //     fgets((char*)&data_uint_2, 2, file_handdle);
    //     std::cout << data_float << "  " << data_uint_1 << "  " << data_uint_2 << std::endl;
    // }

    fclose(file_handdle);
    free(uint_data);
    free(group_data);

    return 0;
}
