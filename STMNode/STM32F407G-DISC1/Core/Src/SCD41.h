/*
 * SCD41.h
 *
 *  Created on: Jan 25, 2022
 *      Author: hunte
 */

#ifndef SRC_SCD41_H_
#define SRC_SCD41_H_

#define SCD41_I2C_ADDRESS 0x62

HAL_StatusTypeDef startPeriodicMeasurment(I2C_HandleTypeDef *hi2x);
HAL_StatusTypeDef readMeasurment(I2C_HandleTypeDef *hi2x, uint16_t * co2, uint16_t *temp, uint16_t *humidity);
HAL_StatusTypeDef getAutomaticSelfCalibration(I2C_HandleTypeDef *hi2x, uint8_t *enabled);

#endif /* SRC_SCD41_H_ */
