#define NUM_TAPS 128

void fir(int input, int *output, int taps[NUM_TAPS])
{
	static int delay_line[NUM_TAPS] = {};

	int result = 0;
	L1: for (int i = NUM_TAPS - 1; i > 0; i--) {
		delay_line[i] = delay_line[i - 1];
	}
	delay_line[0] = input;

	L2: for (int i = 0; i < NUM_TAPS; i++) {
		result += delay_line[i] * taps[i];
	}

	*output = result;
}
