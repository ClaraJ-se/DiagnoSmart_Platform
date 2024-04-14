package com.example.diagnosmart.data.source.repo.impl

import com.example.diagnosmart.data.source.local.patient.LocalPatient
import com.example.diagnosmart.data.source.local.patient.PatientDao
import com.example.diagnosmart.data.source.models.Patient
import com.example.diagnosmart.data.source.models.toExternal
import com.example.diagnosmart.data.source.repo.PatientRepository
import com.example.diagnosmart.di.DefaultDispatcher
import com.example.diagnosmart.utils.ResponseHandler
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import javax.inject.Inject


class PatientRepositoryImpl @Inject constructor(
        private val patientDao: PatientDao,
        @DefaultDispatcher private val dispatcher: CoroutineDispatcher
) : ResponseHandler(), PatientRepository {

    override suspend fun getPatientsStream(): Flow<List<Patient>> {
        TODO("Not yet implemented")
    }

    override suspend fun getMockPatientsStream(): Flow<List<Patient>> {
        return withContext(dispatcher) {
            val patientMockList = listOf(
                    LocalPatient(firstName = "Elsa", lastName = "Andersson", roomNumber = "12"),
                    LocalPatient(firstName = "Gustav", lastName = "Johansson", roomNumber = "13"),
                    LocalPatient(firstName = "Linnea", lastName = "Karlsson", roomNumber = "18"),
                    LocalPatient(firstName = "Axel", lastName = "Eriksson", roomNumber = "16"),
                    LocalPatient(firstName = "Emilia", lastName = "Nilsson", roomNumber = "12"),
                    LocalPatient(firstName = "Viktor", lastName = "Svensson", roomNumber = "14"),
                    LocalPatient(firstName = "Astrid", lastName = "Lindgren", roomNumber = "17"),
                    LocalPatient(firstName = "Henrik", lastName = "Bergqvist", roomNumber = "11"),
                    LocalPatient(firstName = "Elin", lastName = "Sjöberg", roomNumber = "123"),
                    LocalPatient(firstName = "Oscar", lastName = "Ekström", roomNumber = "146"),
                    LocalPatient(firstName = "Matilda", lastName = "Larsson", roomNumber = "143"),
                    LocalPatient(firstName = "Tobias", lastName = "Lundgren", roomNumber = "142")
            )
            flow { emit(patientMockList.toExternal()) }
        }
    }


}