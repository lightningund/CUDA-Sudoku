#include <array>
#include <iostream>
#include <bitset>
#include <vector>
#include <functional>

constexpr size_t board_size = 9;
constexpr size_t total_board = board_size * board_size;

// struct Rule {
// 	std::function<std::vector<Vec2>(const Vec2&)> get_cells;
// 	std::function<bool(const std::vector<uint8_t>&)> is_valid;
// };

// using state_set = std::bitset<board_size>;

// // Checks for no duplicate cells. Used a lot so I just put it here
// bool default_rule(const std::vector<uint8_t>& cell_vals) {
// 	state_set states{};

// 	for (auto& val : cell_vals) {
// 		states.set(val);
// 	}

// 	return states.count() == cell_vals.size();
// };

// const std::vector<Rule> rules = {
// 	{ // Columns
// 		[](const Vec2& pos) -> std::vector<Vec2> {
// 			std::vector<Vec2> cells{};
// 			for (int i{0}; i < BOARD_SIZE; i++) {
// 				if (i != pos.y) cells.push_back(Vec2{pos.x, i});
// 			};

// 			return cells;
// 		}, default_rule
// 	},
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
bool check_state(size_t index, Board& board) {
	return true;
}

__device__
bool check_cell(size_t index, Board& board) {
	if (index > total_board) return true;
	if (board[index] != 255) {
		return check_cell(index + 1, board);
	}

	for (int i = 0; i < board_size; ++i) {
		board[index] = i;

		if (check_state(index, board)) {
			if (check_cell(index + 1, board)) {
				return true;
			}
		}
	}

	board[index] = 255;
	return false;
}

__global__
void solve(Board* board) {
	check_cell(0, *board);
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

	Managed<Board> dev_board{81};
	cudaMemcpy(dev_board.raw, &board, 81, cudaMemcpyHostToDevice);

	print_board(board);

	solve<<<1, 1, 1>>>(dev_board.raw);

	cudaDeviceSynchronize();

	cudaMemcpy(board.data(), dev_board.raw, 81, cudaMemcpyDeviceToHost);

	print_board(board);

	return 0;
}