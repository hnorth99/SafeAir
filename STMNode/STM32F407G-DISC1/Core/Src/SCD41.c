/*
 * SCD41.c
 *
 *  Created on: Jan 25, 2022
 *      Author: hunte
 */

#include "main.h"
#include "SCD41.h"
#include <stdio.h>

HAL_StatusTypeDef getAutomaticSelfCalibration(I2C_HandleTypeDef *hi2x, uint8_t *enabled) {
	HAL_StatusTypeDef ret;
	uint8_t buffer[3];

	// Set buffer contents to getAutomaticSelfCalibration
	buffer[2] = 0x00;
	buffer[1] = 0x23;
	buffer[0] = 0x13;


	// Send command to SCD41
	ret = HAL_I2C_Master_Transmit(hi2x, (SCD41_I2C_ADDRESS<<1), buffer, sizeof(buffer), 100);
	if (ret != HAL_OK) {
		return ret;
	}
	// Read result of command from SCD41
	ret = HAL_I2C_Master_Receive(hi2x, (SCD41_I2C_ADDRESS<<1) | (0x01), buffer, sizeof(buffer), 1);
	if (ret != HAL_OK) {
		return ret;
	}

	enabled = &buffer[0];
	return HAL_OK;
}
