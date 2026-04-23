#ifndef ENCODERS_H
#define ENCODERS_H

#include <Arduino.h>

void IRAM_ATTR leftEncoderISR();
void IRAM_ATTR rightEncoderISR();

#endif // ENCODERS_H
