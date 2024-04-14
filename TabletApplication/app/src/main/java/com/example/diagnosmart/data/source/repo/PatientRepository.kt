package com.example.diagnosmart.data.source.repo

import com.example.diagnosmart.data.source.models.Patient
import com.example.diagnosmart.data.source.models.User
import com.example.diagnosmart.utils.AsyncResult
import kotlinx.coroutines.flow.Flow

interface PatientRepository {

    suspend fun getPatientsStream(): Flow<List<Patient>>

    suspend fun getMockPatientsStream(): Flow<List<Patient>>

}