/*
 * SCD41.h
 *
 *  Created on: Jan 24, 2022
 *      Author: hunte
 */

#ifndef SRC_SCD41_H_
#define SRC_SCD41_H_

#include "main.h"

#define SCD4X_I2C_ADDRESS 0x62

HAL_StatusTypeDef getAutomaticSelfCalibrationEnable(I2C_HandleTypeDef *hi2cX, uint8_t* enabled);

#endif /* SRC_SCD41_H_ */
