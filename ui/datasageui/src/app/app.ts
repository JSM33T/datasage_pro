import { Component } from '@angular/core';
import { RouterLink, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
	selector: 'app-root',
	imports: [
		CommonModule,
		RouterModule,
		RouterLink
	],
	templateUrl: 'app.html',
	styleUrl: 'app.css'
})
export class App {
	logout() {
		throw new Error('Method not implemented.');
	}
}
