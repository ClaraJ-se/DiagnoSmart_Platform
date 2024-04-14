package com.example.diagnosmart.data.source.local.patient

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey
import com.example.diagnosmart.data.source.models.Patient

@Entity(tableName = "patient")
data class LocalPatient(
        @PrimaryKey @ColumnInfo(name = "id") val id: Long = 0,
        val firstName: String? = null,
        val lastName: String? = null,
        val patronymic: String? = null,
        val roomNumber: String? = null,
        val conditions: String? = null,
        val medications: String? = null,
        val familyHistory: String? = null
)

fun LocalPatient.toExternal() = Patient(
        firstName = firstName,
        lastName = lastName,
        patronymic = patronymic,
        roomNumber = roomNumber,
        conditions = conditions,
        medications = medications,
        familyHistory = familyHistory
)
