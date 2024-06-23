#include <array>
#include <iostream>
#include <vector>

#define CUCHECK(str, x) printf(str ": %d\n", (x))

constexpr size_t board_size = 9;
constexpr size_t total_board = board_size * board_size;

struct Rule {
	uint2* (*get_cells)(const uint2);
	bool (*is_valid)(const uint8_t*);
};

// Checks for no duplicate cells. Used a lot so I just put it here
__device__
bool default_rule(const uint8_t* cell_vals) {
	uint16_t states = 0;

	for (int i = 0; i < board_size; ++i) {
		states |= 1 << cell_vals[i];
	}

	return states == ((1 << board_size) - 1);
};

__device__
const Rule column_rule = {
	[](const uint2 pos) -> uint2* {
		uint2* cells;
		CUCHECK("Allocating", cudaMalloc(&cells, board_size));
		for (unsigned int i = 0; i < board_size; ++i) {
			cells[i] = uint2{pos.x, i};
		};

		return cells;
	},
	default_rule
};

// 	{ // Rows
// 		[](const Vec2& pos) -> std::vector<Vec2> {
// 			std::vector<Vec2> cells{};
// 			for (int i{0}; i < BOARD_SIZE; i++) {
// 				if (i != pos.x) cells.push_back(Vec2{i, pos.y});
// 			};

// 			return cells;
// 		}, default_rule
// 	},
// 	{ //Squares
// 		[](const Vec2& pos) -> std::vector<Vec2> {
// 			int square_size{(int)ceil(sqrt(BOARD_SIZE))};
// 			Vec2 square_pos = pos / square_size;
// 			std::vector<Vec2> cells{};
// 			for (int i{0}; i < square_size * square_size; i++) {
// 				Vec2 new_pos{square_pos * square_size};
// 				new_pos += Vec2{(i % square_size), (i / square_size)};
// 				if (new_pos != pos) cells.push_back(new_pos);
// 			}
// 			return cells;
// 		}, default_rule
// 	}
// };

using Board = std::array<uint8_t, total_board>;

__device__
bool check_state(size_t index, uint8_t* board) {
	uint2 pos{(uint16_t)(index % board_size), (uint16_t)(index / board_size)};
	printf("x: %u, y: %u\n", pos.x, pos.y);
	auto cells = column_rule.get_cells(pos);
	uint8_t vals[board_size];

	for (int i = 0; i < board_size; ++i) {
		vals[i] = board[cells[i].x + cells[i].y * board_size];
	}

	CUCHECK("Attempting to free", cudaFree(cells));
	printf("First Val: %hu\n", vals[0]);

	return column_rule.is_valid(vals);
}

__device__
bool check_cell(size_t index, uint8_t* board) {
	if (index > total_board) return true;
	if (board[index] != 255) {
		return check_cell(index + 1, board);
	}

	printf("Gonna check %lu, val: %hu\n", index, board[index]);

	for (uint8_t i = 0; i < (uint8_t)board_size; ++i) {
		printf("Checking val %hu\n", i);
		board[index] = i;

		if (check_state(index, board)) {
			printf("Cell %lu works with val %hu\n", index, i);
			if (check_cell(index + 1, board)) {
				return true;
			}
		} else {
			printf("%hu no worky\n", i);
		}
	}

	printf("welp done\n");

	board[index] = 255;
	return false;
}

__device__
void dev_print_board(uint8_t* board) {
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			uint8_t cell = board[i * 9 + j];
			if (cell == 255) {
				printf("_ ");
			} else {
				printf("%u ", cell + 1);
			}
		}

		printf("\n");
	}

	printf("\n");
}

__global__
void solve(uint8_t* board) {
	printf("Gonna solve it\n");
	dev_print_board(board);
	check_cell(0, board);
}

void print_board(Board& board) {
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			uint8_t cell = board[i * 9 + j];
			if (cell == 255) {
				std::cout << "_";
			} else {
				std::cout << (int)cell + 1;
			}

			std::cout << " ";
		}

		std::cout << "\n";
	}

	std::cout << "\n";
}

// Wrapper for managed memory objects
template <typename T>
struct Managed {
	T* raw;

	// Takes the size as a number of bytes
	Managed(size_t size) {
		cudaMallocManaged(&raw, size);
	}

	~Managed() {
		cudaFree(raw);
	}
};

int main() {
	std::string input =
		"010000504"
		"096007000"
		"000200010"
		"000000807"
		"085060002"
		"004000000"
		"030000090"
		"009030005"
		"000540060";

	Board board;

	int loop = 0;
	for (char c : input) {
		if (c != '0') {
			uint8_t val = c - '1';
			board[loop] = val;
		} else {
			board[loop] = 255;
		}

		loop++;
	}

	Managed<uint8_t> dev_board{81};
	CUCHECK("Attempting to copy", cudaMemcpy(dev_board.raw, board.data(), 81, cudaMemcpyHostToDevice));

	print_board(board);

	solve<<<1, 1, 1>>>(dev_board.raw);

	CUCHECK("Attempting to sync", cudaDeviceSynchronize());

	CUCHECK("Attempting to copy back", cudaMemcpy(board.data(), dev_board.raw, 81, cudaMemcpyDeviceToHost));

	print_board(board);

	return 0;
}