package dev.sruti.bench

import android.content.Context
import android.os.BatteryManager
import android.os.PowerManager

/**
 * Samples the signals that actually bound sustained inference on a phone.
 *
 * Thermal headroom is the interesting one: sustained decode pulls several watts
 * and the SoC will throttle within minutes. Reading status lets the app back off
 * on its own terms — degrading token rate deliberately — instead of having the
 * kernel do it abruptly mid-response.
 */
class ThermalMonitor(context: Context) {

    private val powerManager =
        context.getSystemService(Context.POWER_SERVICE) as PowerManager

    private val batteryManager =
        context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    data class Sample(
        val elapsedMillis: Long,
        val thermalStatus: Int,
        val batteryPercent: Int,
        /** Instantaneous current in microamps; negative means discharging. */
        val currentMicroAmps: Long,
        /** Remaining charge in nanowatt-hours, or -1 when unavailable. */
        val chargeNanoWattHours: Long,
    ) {
        val thermalStatusName: String
            get() = thermalStatusName(thermalStatus)
    }

    fun sample(elapsedMillis: Long): Sample = Sample(
        elapsedMillis = elapsedMillis,
        thermalStatus = powerManager.currentThermalStatus,
        batteryPercent = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY),
        currentMicroAmps = batteryManager
            .getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW),
        chargeNanoWattHours = batteryManager
            .getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER),
    )

    companion object {
        fun thermalStatusName(status: Int): String = when (status) {
            PowerManager.THERMAL_STATUS_NONE -> "none"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
            else -> "unknown($status)"
        }
    }
}
