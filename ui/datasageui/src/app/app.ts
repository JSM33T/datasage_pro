import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
	selector: 'app-root',
	imports: [
		CommonModule,
		RouterModule
	],
	templateUrl: 'app.html',
	styleUrl: 'app.css'
})
export class App {
	logout() {
		throw new Error('Method not implemented.');
	}
}
