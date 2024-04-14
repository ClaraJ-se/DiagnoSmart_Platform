package com.example.diagnosmart.data.source.models

import com.example.diagnosmart.data.source.local.patient.LocalPatient

data class Patient(
        val firstName: String? = "",
        val lastName: String? = "",
        val patronymic: String? = "",
        val roomNumber: String? = "",
        val conditions: String? = "",
        val medications: String? = "",
        val familyHistory: String? = ""
)

fun Patient.toLocal() = LocalPatient(
        firstName = firstName ?: "",
        lastName = lastName ?: "",
        patronymic = patronymic ?: "",
        roomNumber = roomNumber ?: "",
        conditions = conditions ?: "",
        medications = medications ?: "",
        familyHistory = familyHistory ?: ""
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

fun List<LocalPatient>.toExternal() = map(LocalPatient::toExternal)
